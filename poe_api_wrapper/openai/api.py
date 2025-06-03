from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from daphne.cli import CommandLineInterface
from typing import Any, Dict, Tuple, Union, AsyncGenerator
from poe_api_wrapper import AsyncPoeApi
from poe_api_wrapper.openai import helpers
from poe_api_wrapper.openai.type import *
import orjson, asyncio, random, os, uuid, hashlib
from httpx import AsyncClient
from typing import List, Dict, Tuple, Optional, Any, Union # Added Any for message content & Union

# Forward declaration for AsyncPoeApi type hint in PromptCacheManager
if False: # This block is not executed, for type hinting only
    from poe_api_wrapper import AsyncPoeApi

class PromptCacheManager:
    # Cache store: key -> {"chat_id": str, "client": AsyncPoeApi, "subscription": bool}
    def __init__(self, cache_store: Dict[str, Dict[str, Union[str, 'AsyncPoeApi', bool]]], logging_enabled: bool):
        self.cache_store = cache_store
        self.logging_enabled = logging_enabled
        if self.logging_enabled:
            print(f"[CACHE_MANAGER_INIT] PromptCacheManager initialized. Current cache size: {len(self.cache_store)}")

    def _log(self, message: str):
        if self.logging_enabled:
            print(message)

    def _generate_cache_key(self, content: Any) -> Optional[str]:
        key_string = None
        content_type_str = str(type(content))

        if isinstance(content, str):
            if content: # Ensure string is not empty
                key_string = content
            else:
                self._log(f"[CACHE_MANAGER_KEYGEN_SKIP] Content is an empty string. Skipping key generation.")
                return None
        elif isinstance(content, (list, dict)): # Handle list or dict content types (e.g. for multimodal)
            if not content: # Ensure list/dict is not empty
                self._log(f"[CACHE_MANAGER_KEYGEN_SKIP] Content is an empty list/dict. Skipping key generation.")
                return None
            try:
                # Sort keys for dictionaries to ensure canonical representation if order can vary
                # For lists of dicts, orjson.dumps should be consistent if underlying dicts are consistent
                # Option ORJSON_OPT_SORT_KEYS can be used if needed, but default may be sufficient
                serialized_content = orjson.dumps(content)
                key_string = serialized_content.decode('utf-8')
                key_string = key_string[:300]
            except Exception as e:
                self._log(f"[CACHE_MANAGER_KEYGEN_SERIALIZE_FAIL] Failed to serialize {content_type_str} content for key: {e}. Content preview: '{str(content)[:150]}...'")
                return None
        elif content: # Fallback for other non-empty, non-string, non-list/dict types
             # This might not produce a canonical key for complex objects, but provides a basic string form.
            self._log(f"[CACHE_MANAGER_KEYGEN_WARN] Content type {content_type_str} is not str/list/dict. Attempting str() conversion for key. Preview: '{str(content)[:100]}...'")
            key_string = str(content)


        if key_string: # Check if key_string was successfully set and is not empty
            try:
                key_bytes = key_string.encode('utf-8')
                key = hashlib.md5(key_bytes).hexdigest()
                self._log(f"[CACHE_MANAGER_KEYGEN] Generated cache key: {key} from content (type: {content_type_str}, first 100 chars of serialized_form): '{key_string[:100]}...'")
                return key
            except Exception as e:
                self._log(f"[CACHE_MANAGER_KEYGEN_HASH_FAIL] Failed to hash string_repr for key: {e}. String preview: '{key_string[:150]}...'")
                return None
            
        self._log(f"[CACHE_MANAGER_KEYGEN_FAIL] Failed to generate key, content not suitable or empty. Original content type: {content_type_str}, preview: '{str(content)[:150]}...'")
        return None

    def check_cache(self, messages: List[Dict[str, Any]]) -> Tuple[Optional['AsyncPoeApi'], Optional[bool], Optional[str], List[Dict[str, Any]], bool, Optional[str]]:
        """
        Checks the cache for a given set of messages.
        Returns: (cached_client, cached_subscription, chat_id_to_use, processed_messages, chat_was_reused, cache_key_for_storage)
        'chat_was_reused' primarily indicates if the message content key matched and implies message list might be trimmed.
        """
        cached_client: Optional['AsyncPoeApi'] = None
        cached_subscription: Optional[bool] = None
        chat_id_to_use: Optional[str] = None
        
        chat_was_reused = False # True if key matches and implies potential message trimming / chat_id reuse
        cache_key_for_storage = None
        processed_messages = list(messages) # Work with a copy

        if len(messages) >= 2:
            content_for_key_message = messages[1]
            content_for_key = content_for_key_message.get('content', '')
            cache_key_for_storage = self._generate_cache_key(content_for_key)

            if cache_key_for_storage and cache_key_for_storage in self.cache_store:
                cached_entry = self.cache_store[cache_key_for_storage]
                ret_client = cached_entry.get("client")
                ret_subscription = cached_entry.get("subscription")
                ret_chat_id = cached_entry.get("chat_id")

                # A "full hit" for client reuse requires client, subscription, and chat_id
                if ret_client is not None and ret_subscription is not None and ret_chat_id is not None:
                    # Type assertion for linters/mypy if they struggle with Union from dict.get()
                    cached_client = ret_client # type: ignore
                    cached_subscription = ret_subscription # type: ignore
                    chat_id_to_use = ret_chat_id # type: ignore
                    self._log(f"[CACHE_MANAGER_HIT_FULL] FULL CACHE HIT: Using cached client, subscription, and chatId '{chat_id_to_use}' for key {cache_key_for_storage}.")
                else: # Partial hit (e.g. key exists but data incomplete), treat as client/sub miss
                     self._log(f"[CACHE_MANAGER_HIT_PARTIAL] PARTIAL CACHE HIT: Key {cache_key_for_storage} found, but client/sub/chat_id incomplete. Will proceed as client miss.")
                     # chat_id_to_use might still be set if ret_chat_id was valid, even if client/sub was not
                     if ret_chat_id is not None:
                         chat_id_to_use = ret_chat_id # type: ignore

                # Regardless of full/partial client hit, if key matched, messages are processed for chat_id reuse context
                if processed_messages:
                    original_message_count = len(processed_messages)
                    last_message = processed_messages[-1]
                    processed_messages = [last_message]
                    chat_was_reused = True # This signifies that messages were trimmed due to key match.
                    self._log(f"[CACHE_MANAGER_MSG_TRIM] Messages for key {cache_key_for_storage} trimmed from {original_message_count} to 1.")

            elif cache_key_for_storage:
                self._log(f"[CACHE_MANAGER_MISS] CACHE MISS (initial check): Key {cache_key_for_storage} not found in cache store.")
        else:
            self._log(f"[CACHE_MANAGER_SKIP] Skipping cache check: Not enough messages for key generation (count: {len(messages)}).")
        
        return cached_client, cached_subscription, chat_id_to_use, processed_messages, chat_was_reused, cache_key_for_storage

    def store_in_cache(self, cache_key: Optional[str], chat_id: Optional[str],
                       client_instance: 'AsyncPoeApi', subscription_status: bool,
                       bot_name: str, source: str):
        if not cache_key:
            self._log(f"[CACHE_MANAGER_STORE_FAIL:{source.upper()}] No cache key provided for bot '{bot_name}'. Cannot store.")
            return
        if not chat_id:
            self._log(f"[CACHE_MANAGER_STORE_FAIL:{source.upper()}] No (or empty) chat_id provided for bot '{bot_name}' with key '{cache_key}'. Cannot store.")
            return
        if client_instance is None: # Also check client_instance
             self._log(f"[CACHE_MANAGER_STORE_FAIL:{source.upper()}] No client_instance provided for bot '{bot_name}' with key '{cache_key}'. Cannot store.")
             return
            
        self.cache_store[cache_key] = {
            "chat_id": chat_id,
            "client": client_instance,
            "subscription": subscription_status
        }
        self._log(f"[CACHE_MANAGER_STORE:{source.upper()}] Stored chatId '{chat_id}', client instance, and subscription ({subscription_status}) for key '{cache_key}' (bot: {bot_name}). Cache size: {len(self.cache_store)}")


DIR = os.path.dirname(os.path.abspath(__file__))

# --- BEGIN CACHING LOGGING FLAG ---
# Read from environment variable, default to "false" if not set
POE_API_CACHE_LOGGING_ENABLED_STR = os.environ.get("POE_API_CACHE_LOGGING_ENABLED", "false")
CACHE_LOGGING_ENABLED = POE_API_CACHE_LOGGING_ENABLED_STR.lower() == "true"
if CACHE_LOGGING_ENABLED:
    print(f"[CACHE_INIT] Prompt caching logging is ENABLED (POE_API_CACHE_LOGGING_ENABLED='{POE_API_CACHE_LOGGING_ENABLED_STR}')")
else:
    print(f"[CACHE_INIT] Prompt caching logging is DISABLED (POE_API_CACHE_LOGGING_ENABLED='{POE_API_CACHE_LOGGING_ENABLED_STR}')")
# --- END CACHING LOGGING FLAG ---

app = FastAPI(title="Poe API Wrapper", description="OpenAI Proxy Server")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

with open(os.path.join(DIR, "secrets.json"), "rb") as f:
    TOKENS = orjson.loads(f.read())
    if "tokens" not in TOKENS:
        raise Exception("Tokens not found in secrets.json")
    app.state.tokens = TOKENS["tokens"]

with open(os.path.join(DIR, "models.json"), "rb") as f:
    models = orjson.loads(f.read())
    app.state.models = models
app.state.chat_id_cache = {}

# Initialize PromptCacheManager and store on app.state
app.state.prompt_cache_manager = PromptCacheManager(
    cache_store=app.state.chat_id_cache,
    logging_enabled=CACHE_LOGGING_ENABLED # Assumes CACHE_LOGGING_ENABLED is defined globally
)

async def call_tools(messages, tools, tool_choice):
    response = await message_handler("gpt4_o_mini", messages, 128000, tools, tool_choice)
    tool_calls = None
    client, _ = await rotate_token(app.state.tokens)
    async for chunk in client.send_message(bot="gpt4_o_mini", message=response["message"]):
        try:
            res_list = orjson.loads(chunk["text"].strip().replace("\n", "").replace("\\",""))
            if res_list and type(res_list) == list:
                tool_calls = res_list
                break
        except Exception as e:
            pass
        
    return tool_calls
    
    
@app.get("/", response_model=None)
async def index() -> ORJSONResponse:
    return ORJSONResponse({"message": "Welcome to Poe Api Wrapper reverse proxy!",
                            "docs": "See project docs @ https://github.com/snowby666/poe-api-wrapper"})


@app.api_route("/models/{model}", methods=["GET", "POST", "PUT", "PATCH", "HEAD"], response_model=None)
@app.api_route("/models/{model}", methods=["GET", "POST", "PUT", "PATCH", "HEAD"], response_model=None)
@app.api_route("/models", methods=["GET", "POST", "PUT", "PATCH", "HEAD"], response_model=None)
@app.api_route("/v1/models", methods=["GET", "POST", "PUT", "PATCH", "HEAD"], response_model=None)
async def list_models(request: Request, model: str = None) -> ORJSONResponse:
    if model:
        if model not in app.state.models:
            raise HTTPException(detail={"error": {"message": "Invalid model.", "type": "error", "param": None, "code": 400}}, status_code=400)
        return ORJSONResponse({"id": model, "object": "model", "created": await helpers.__generate_timestamp(), "owned_by": app.state.models[model]["owned_by"], "tokens": app.state.models[model]["tokens"], "endpoints": app.state.models[model]["endpoints"]})
    modelsData = [{"id": model, "object": "model", "created": await helpers.__generate_timestamp(), "owned_by": values["owned_by"], "tokens": values["tokens"], "endpoints": values["endpoints"]} for model, values in app.state.models.items()]
    return ORJSONResponse({"object": "list", "data": modelsData})


@app.api_route("/chat/completions", methods=["POST", "OPTIONS"], response_model=None)
@app.api_route("/v1/chat/completions", methods=["POST", "OPTIONS"], response_model=None)
async def chat_completions(request: Request, data: ChatData) -> Union[StreamingResponse, ORJSONResponse]:
    messages_data, model, streaming, max_tokens, stream_options, tools, tool_choice = data.messages, data.model, data.stream, data.max_tokens, data.stream_options, data.tools, data.tool_choice

    # --- BEGIN CACHING LOGIC ---
    cached_client, cached_subscription, chat_id_to_use, messages, chat_was_reused, cache_key_for_storage = \
        app.state.prompt_cache_manager.check_cache(messages_data)
    # `messages` variable is updated by check_cache if key matched (chat_was_reused is True)
    # --- END CACHING LOGIC ---

    # Validate messages format
    if not await helpers.__validate_messages_format(messages): # `messages` might be trimmed here
        raise HTTPException(detail={"error": {"message": "Invalid messages format.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if model not in app.state.models:
        raise HTTPException(detail={"error": {"message": "Invalid model.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if tools and len(tools) > 20:
        raise HTTPException(detail={"error": {"message": "Maximum 20 tools are allowed.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    include_usage = stream_options.get("include_usage", False) if stream_options else False
    
    modelData = app.state.models[model]
    baseModel, tokensLimit, endpoints, premiumModel = modelData["baseModel"], modelData["tokens"], modelData["endpoints"], modelData["premium_model"]
    
    if "/v1/chat/completions" not in endpoints:
        raise HTTPException(detail={"error": {"message": "This model does not support chat completions.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    # Determine client and subscription to use
    client_to_use: Optional[AsyncPoeApi] = None
    subscription_to_use: Optional[bool] = None

    # A "full cache hit for client reuse" means we have a cached client and subscription,
    # and chat_was_reused is True indicating the primary prompt key matched.
    if cached_client and cached_subscription is not None and chat_was_reused: # Re-evaluate condition based on new `check_cache`
        client_to_use = cached_client
        subscription_to_use = cached_subscription
        if CACHE_LOGGING_ENABLED: # CACHE_LOGGING_ENABLED assumed to be global
             app.state.prompt_cache_manager._log(f"[CHAT_COMPLETIONS] Using cached client and subscription for key {cache_key_for_storage}.")
    else:
        if CACHE_LOGGING_ENABLED:
            log_reason = "no cached client/subscription"
            if not chat_was_reused:
                log_reason = "cache key did not match (chat_was_reused is False)"
            elif not cached_client:
                 log_reason = "no cached client"
            elif cached_subscription is None:
                 log_reason = "no cached subscription status"

            app.state.prompt_cache_manager._log(f"[CHAT_COMPLETIONS] Rotating token: {log_reason}.")
        client_to_use, subscription_to_use = await rotate_token(app.state.tokens)

    if client_to_use is None: # Should not happen if rotate_token works, but as a safeguard
        raise HTTPException(status_code=500, detail="Failed to obtain a Poe client.")

    if premiumModel and not subscription_to_use:
        raise HTTPException(detail={"error": {"message": "Premium model requires a subscription.", "type": "error", "param": None, "code": 402}}, status_code=402)
    
    text_messages, image_urls = await helpers.__split_content(messages) # Use potentially trimmed `messages`
    
    # message_handler needs the full untrimmed messages for context summarization if it's a fresh request
    # If messages were trimmed by cache, message_handler will receive only the last user message.
    # This is correct behavior for Poe, which relies on server-side context for reused chat_id.
    response_dict = await message_handler(baseModel, text_messages, tokensLimit) # Pass text_messages
    prompt_tokens = await helpers.__tokenize(''.join([str(message) for message in response_dict["message"]]))
    
    if prompt_tokens > tokensLimit:
        raise HTTPException(detail={"error": {"message": f"Your prompt exceeds the maximum context length of {tokensLimit} tokens.", "type": "error", "param": None, "code": 400}}, status_code=400)
        
    if max_tokens and sum((max_tokens, prompt_tokens)) > tokensLimit:
        raise HTTPException(detail={"error": {
                                        "message": f"This model's maximum context length is {tokensLimit} tokens. However your request exceeds this limit ({max_tokens} in max_tokens, {prompt_tokens} in messages).", 
                                        "type": "error", 
                                        "param": None, 
                                        "code": 400}
                                    }, status_code=400)
    
    raw_tool_calls = None
    if tools:
        if not tool_choice:
            tool_choice = "auto"
        raw_tool_calls = await call_tools(messages, tools, tool_choice)
    
    if raw_tool_calls:
        response = {"bot": "gpt4_o_mini", "message": ""}
        prompt_tokens = await helpers.__tokenize(''.join([str(message["content"]) for message in text_messages]))
        
    completion_id = await helpers.__generate_completion_id()
    
    # Important: Pass the client_to_use and subscription_to_use for potential cache storage downstream
    # The `response_dict` contains the bot name and formatted message for Poe.
    return await streaming_response(client_to_use, response_dict, model, completion_id, prompt_tokens, image_urls, max_tokens, include_usage, raw_tool_calls,
                                    chat_id_to_use=chat_id_to_use,
                                    chat_was_reused=chat_was_reused, # This signals if original prompt matched
                                    cache_key_to_store=cache_key_for_storage,
                                    # Pass the client and subscription actually used for this request.
                                    # These will be stored in cache if it's a miss and a new entry is created.
                                    client_for_cache_storage=client_to_use,
                                    subscription_for_cache_storage=subscription_to_use
                                    ) \
        if streaming else await non_streaming_response(client_to_use, response_dict, model, completion_id, prompt_tokens, image_urls, max_tokens, raw_tool_calls,
                                                       chat_id_to_use=chat_id_to_use,
                                                       chat_was_reused=chat_was_reused,
                                                       cache_key_to_store=cache_key_for_storage,
                                                       client_for_cache_storage=client_to_use,
                                                       subscription_for_cache_storage=subscription_to_use
                                                       )


@app.api_route("/images/generations", methods=["POST", "OPTIONS"], response_model=None)
@app.api_route("/v1/images/generations", methods=["POST", "OPTIONS"], response_model=None)
async def create_images(request: Request, data: ImagesGenData) -> ORJSONResponse:
    prompt, model, n, size = data.prompt, data.model, data.n, data.size
    
    if not isinstance(prompt, str):
        raise HTTPException(detail={"error": {"message": "Invalid prompt.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if model not in app.state.models:
        raise HTTPException(detail={"error": {"message": "Invalid model.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if not isinstance(n, int) or n < 1:
        raise HTTPException(detail={"error": {"message": "Invalid n value.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if size == "1024x1024":
        aspect_ratio = ""
    elif "sizes" in app.state.models[model] and size in app.state.models[model]["sizes"]:
        aspect_ratio = app.state.models[model]["sizes"][size]
    else:
        raise HTTPException(detail={"error": {"message": f"Invalid size for {model}. Available sizes: {', '.join(app.state.models[model]['sizes']) if 'sizes' in app.state.models[model] else '1024x1024'}", "type": "error", "param": None, "code": 400}}, status_code=400)

    modelData = app.state.models[model]
    baseModel, tokensLimit, endpoints, premiumModel = modelData["baseModel"], modelData["tokens"], modelData["endpoints"], modelData["premium_model"]
    
    if "/v1/images/generations" not in endpoints:
        raise HTTPException(detail={"error": {"message": "This model does not support image generation.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    client, subscription = await rotate_token(app.state.tokens)
    
    if premiumModel and not subscription:
        raise HTTPException(detail={"error": {"message": "Premium model requires a subscription.", "type": "error", "param": None, "code": 402}}, status_code=402)
    
    response = await image_handler(baseModel, prompt, tokensLimit)
    
    urls = []
    for _ in range(n):
        image_generation = await generate_image(client, response, aspect_ratio)
        urls.extend([url for url in image_generation.split() if url.startswith("https://")])
        if len(urls) >= n:
            break
    urls = urls[-n:]
    
    if len(urls) == 0:
        raise HTTPException(detail={"error": {"message": f"The provider for {model} sent an invalid response.", "type": "error", "param": None, "code": 500}}, status_code=500)
        
    async with AsyncClient(http2=True) as fetcher:
        for url in urls:
            r = await fetcher.get(url)
            content_type = r.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                raise HTTPException(detail={"error": {"message": "The content returned was not an image.", "type": "error", "param": None, "code": 500}}, status_code=500)

    return ORJSONResponse({"created": await helpers.__generate_timestamp(), "data": [{"url": url} for url in urls]})


@app.api_route("/images/edits", methods=["POST", "OPTIONS"], response_model=None)
@app.api_route("/v1/images/edits", methods=["POST", "OPTIONS"], response_model=None)
async def edit_images(request: Request, data: ImagesEditData) -> ORJSONResponse:
    image, prompt, model, n, size = data.image, data.prompt, data.model, data.n, data.size
    
    if not (isinstance(image, str) and (os.path.exists(image) or image.startswith("http"))):
        raise HTTPException(detail={"error": {"message": "Invalid image.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if not isinstance(prompt, str):
        raise HTTPException(detail={"error": {"message": "Invalid prompt.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if model not in app.state.models:
        raise HTTPException(detail={"error": {"message": "Invalid model.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if not isinstance(n, int) or n < 1:
        raise HTTPException(detail={"error": {"message": "Invalid n value.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    if size == "1024x1024":
        aspect_ratio = ""
    elif "sizes" in app.state.models[model] and size in app.state.models[model]["sizes"]:
        aspect_ratio = app.state.models[model]["sizes"][size]
    else:
        raise HTTPException(detail={"error": {"message": f"Invalid size for {model}. Available sizes: {', '.join(app.state.models[model]['sizes']) if 'sizes' in app.state.models[model] else '1024x1024'}", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    modelData = app.state.models[model]
    baseModel, tokensLimit, endpoints, premiumModel = modelData["baseModel"], modelData["tokens"], modelData["endpoints"], modelData["premium_model"]
    
    if "/v1/images/edits" not in endpoints:
        raise HTTPException(detail={"error": {"message": "This model does not support image editing.", "type": "error", "param": None, "code": 400}}, status_code=400)
    
    client, subscription = await rotate_token(app.state.tokens)
    
    if premiumModel and not subscription:
        raise HTTPException(detail={"error": {"message": "Premium model requires a subscription.", "type": "error", "param": None, "code": 402}}, status_code=402)
    
    response = await image_handler(baseModel, prompt, tokensLimit)
    
    urls = []
    for _ in range(n):
        image_generation = await generate_image(client, response, aspect_ratio, [image])
        urls.extend([url for url in image_generation.split() if url.startswith("https://")])
        if len(urls) >= n:
            break
    urls = urls[-n:]
        
    if len(urls) == 0:
        raise HTTPException(detail={"error": {"message": f"The provider for {model} sent an invalid response.", "type": "error", "param": None, "code": 500}}, status_code=500)
    
    async with AsyncClient(http2=True) as fetcher:
        for url in urls:
            r = await fetcher.get(url)
            content_type = r.headers.get("Content-Type", "")
            if not content_type.startswith("image/"):
                raise HTTPException(detail={"error": {"message": "The content returned was not an image.", "type": "error", "param": None, "code": 500}}, status_code=500)
            
    return ORJSONResponse({"created": await helpers.__generate_timestamp(), "data": [{"url": url} for url in urls]})
   

async def image_handler(baseModel: str, prompt: str, tokensLimit: int) -> dict:
    try:
        message = await helpers.__progressive_summarize_text(prompt, min(len(prompt), tokensLimit))
        return {"bot": baseModel, "message": message}
    except Exception as e:
        raise HTTPException(detail={"error": {"message": f"Failed to truncate prompt. Error: {e}", "type": "error", "param": None, "code": 400}}, status_code=400) from e
   
   
async def message_handler(
    baseModel: str, messages: List[Dict[str, str]], tokensLimit: int, tools: list[dict[str, str]] = None, tool_choice = None
) -> dict:
    
    try:
        main_request = messages[-1]["content"]
        check_user = messages[::-1]
        for message in check_user:
            if message["role"] == "user":
                main_request = message["content"]
                break
        
        if tools:
            rest_tools = await helpers.__convert_functions_format(tools, tool_choice)
            if messages[0]["role"] == "system":
                messages[0]["content"] += rest_tools
            else:
                messages.insert(0, {"role": "system", "content": rest_tools})

        full_string = await helpers.__stringify_messages(messages=messages)
        
        history_string = await helpers.__stringify_messages(messages=messages[:-1])

        full_tokens = await helpers.__tokenize(full_string)
        
        if full_tokens > tokensLimit:
            history_string = await helpers.__progressive_summarize_text(
                history_string, tokensLimit - await helpers.__tokenize(main_request) - 100
            )
        
        message = f"Your current message context: \n{history_string}\n\nReply to most recent message: {main_request}\n\n"
        return {"bot": baseModel, "message": message}
    except Exception as e:
        raise HTTPException(detail={"error": {"message": f"Failed to process messages. Error: {e}", "type": "error", "param": None, "code": 400}}, status_code=400) from e


async def generate_image(client: AsyncPoeApi, response: dict, aspect_ratio: str, image: list = []) -> str:
    try:
        async for chunk in client.send_message(bot=response["bot"], message=f"{response['message']} {aspect_ratio}", file_path=image):
            pass
        return chunk["text"]
    except Exception as e:
        raise HTTPException(detail={"error": {"message": f"Failed to generate image. Error: {e}", "type": "error", "param": None, "code": 500}}, status_code=500) from e
    
    
async def create_completion_data(
    completion_id: str, created: int, model: str, chunk: str = None, 
    finish_reason: str = None, include_usage: bool=False,
    prompt_tokens: int = 0, completion_tokens: int = 0, raw_tool_calls: list[dict[str, str]] = None 
) -> Dict[str, Union[str, list, float]]:
    
    completion_data = ChatCompletionChunk(
        id=f"chatcmpl-{completion_id}",
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[
            ChatCompletionChunkChoice(
                index=0,
                delta=MessageResponse(
                    role="assistant", 
                    content=chunk,
                    tool_calls=[ChoiceDeltaToolCall(
                            index = raw_tool_calls.index(tool_call),
                            id=f"call-{await helpers.__generate_completion_id()}",
                            function=ChoiceDeltaToolCallFunction(name=tool_call["name"], arguments=orjson.dumps(tool_call["arguments"]))) for tool_call in raw_tool_calls] if raw_tool_calls else None
                    ),
                finish_reason=finish_reason,
            )
        ],
    )
    
    if include_usage:
        completion_data.usage = None
        if finish_reason in ("stop", "length"):
            completion_data.usage = ChatCompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens)
    
    return completion_data.model_dump()
    
    
async def generate_chunks(
    client: AsyncPoeApi, response: dict, model: str, completion_id: str,
    prompt_tokens: int, image_urls: List[str], max_tokens: int, include_usage:bool, raw_tool_calls: list[dict[str, str]] = None,
    chat_id_to_use: str = None, # This is the chatId to *use* for send_message if available
    chat_was_reused: bool = False, # True if messages were trimmed due to primary key match
    cache_key_to_store: str = None, # The key under which new cache entry will be stored if it's a miss
    client_for_cache_storage: Optional[AsyncPoeApi] = None, # The client instance that generated the response
    subscription_for_cache_storage: Optional[bool] = None # The subscription status of that client
) -> AsyncGenerator[bytes, None]:
    
    bot_name = response["bot"]
    # To store a new cache entry, we need the client that processed this request and its subscription.
    # This is passed via client_for_cache_storage and subscription_for_cache_storage.
    # The 'client' parameter for this function is the one actually used to send the message.
    # For the first message in a new conversation (cache miss for client), 'client' and 'client_for_cache_storage' will be the same.

    first_chunk_processed_for_cache_store = False # To ensure store_in_cache is called only once for this request
    try:
        completion_timestamp = await helpers.__generate_timestamp()
        finish_reason = "stop"
        
        if not raw_tool_calls:
            # 'client' here is client_to_use from chat_completions (either cached or newly rotated)
            async for chunk in client.send_message(bot=bot_name, message=response["message"], file_path=image_urls, chatId=chat_id_to_use):
                chunk_token = await helpers.__tokenize(chunk["text"])
                
                if max_tokens and chunk_token >= max_tokens:
                    await client.cancel_message(chunk)
                    finish_reason = "length"
                    break
                
                # --- BEGIN STORE CLIENT & CHAT_ID ON CACHE MISS ---
                # If chat_was_reused is False, it means this is a new conversation context based on prompt key.
                # Thus, we need to store the client, its sub status, and the new chat_id.
                if not chat_was_reused and cache_key_to_store and bot_name and not first_chunk_processed_for_cache_store:
                    retrieved_chat_id = client.last_chat_id # Chat ID generated by the send_message call
                    if retrieved_chat_id and client_for_cache_storage is not None and subscription_for_cache_storage is not None:
                        app.state.prompt_cache_manager.store_in_cache(
                            cache_key=cache_key_to_store,
                            chat_id=retrieved_chat_id,
                            client_instance=client_for_cache_storage, # Store the client that handled this request
                            subscription_status=subscription_for_cache_storage, # Store its subscription status
                            bot_name=bot_name,
                            source="stream"
                        )
                        # client.last_chat_id is typically read-only or managed internally after a send.
                        # If it needs to be cleared, it should be handled by the PoeApi library or a wrapper method.
                        # For now, assume `last_chat_id` reflects the ID of the *just completed* `send_message`.
                        # Mark that we've attempted to store this for this request
                        first_chunk_processed_for_cache_store = True
                    elif CACHE_LOGGING_ENABLED:
                        app.state.prompt_cache_manager._log(f"[GENERATE_CHUNKS_CACHE_STORE_SKIP:STREAM] Could not store. Missing retrieved_chat_id ({retrieved_chat_id is not None}), client_for_cache_storage ({client_for_cache_storage is not None}), or subscription_for_cache_storage ({subscription_for_cache_storage is not None}).")
                # --- END STORE CLIENT & CHAT_ID ON CACHE MISS ---
                
                content = await create_completion_data(
                                                    completion_id=completion_id,
                                                    created=completion_timestamp,
                                                    model=model,
                                                    chunk=chunk["response"],
                                                    include_usage=include_usage
                                                    )
                
                yield b"data: " + orjson.dumps(content) + b"\n\n"
                await asyncio.sleep(0.001)
                
            end_completion_data = await create_completion_data(
                                                            completion_id=completion_id,
                                                            created=completion_timestamp,
                                                            model=model,
                                                            finish_reason=finish_reason,
                                                            include_usage=include_usage,
                                                            prompt_tokens=prompt_tokens,
                                                            completion_tokens=chunk_token if 'chunk_token' in locals() else 0 # Handle case where loop might not run
                                                            )
            
            yield b"data: " +  orjson.dumps(end_completion_data) + b"\n\n"
            
        else:
            chunk_token = await helpers.__tokenize(''.join([str(tool_call["name"]) + str(tool_call["arguments"]) for tool_call in raw_tool_calls]))
            content = await create_completion_data(
                                                completion_id=completion_id, 
                                                created=completion_timestamp,
                                                model=model, 
                                                chunk=None,
                                                finish_reason="tool_calls",
                                                include_usage=include_usage,
                                                prompt_tokens=prompt_tokens,
                                                completion_tokens=chunk_token,
                                                raw_tool_calls=raw_tool_calls)
            yield b"data: " + orjson.dumps(content) + b"\n\n"
            await asyncio.sleep(0.01)
   
        yield b"data: [DONE]\n\n"
    except GeneratorExit:
        pass
    except Exception as e:
        raise HTTPException(detail={"error": {"message": f"Failed to stream response. Error: {e}", "type": "error", "param": None, "code": 500}}, status_code=500) from e

    
async def streaming_response(
    client: AsyncPoeApi, # This is the client_to_use determined in chat_completions
    response_content: dict, # Renamed from 'response' to avoid confusion with FastAPI 'response'
    model_name: str, # Renamed from 'model'
    completion_id: str,
    prompt_tokens: int,
    image_urls: List[str],
    max_tokens: int,
    include_usage: bool,
    raw_tool_calls: list[dict[str, str]] = None,
    chat_id_to_use: str = None,
    chat_was_reused: bool = False,
    cache_key_to_store: str = None,
    client_for_cache_storage: Optional[AsyncPoeApi] = None, # New param
    subscription_for_cache_storage: Optional[bool] = None # New param
) -> StreamingResponse:
    
    # Pass all relevant params, including the client/sub to store on miss
    return StreamingResponse(content=generate_chunks(
                                client, response_content, model_name, completion_id, prompt_tokens,
                                image_urls, max_tokens, include_usage, raw_tool_calls,
                                chat_id_to_use, chat_was_reused, cache_key_to_store,
                                client_for_cache_storage, subscription_for_cache_storage # Pass down
                                ),
                             status_code=200,
                             headers={"X-Request-ID": str(uuid.uuid4()), "Content-Type": "text/event-stream"})


async def non_streaming_response(
    client: AsyncPoeApi, # This is the client_to_use determined in chat_completions
    response_content: dict, # Renamed from 'response'
    model_name: str, # Renamed from 'model'
    completion_id: str,
    prompt_tokens: int,
    image_urls: List[str],
    max_tokens: int,
    raw_tool_calls: list[dict[str, str]] = None,
    chat_id_to_use: str = None,
    chat_was_reused: bool = False,
    cache_key_to_store: str = None,
    client_for_cache_storage: Optional[AsyncPoeApi] = None, # New param
    subscription_for_cache_storage: Optional[bool] = None # New param
) -> ORJSONResponse:
    
    bot_name = response_content["bot"] # Use response_content
    chunk_text_for_tokens = ""
    if not raw_tool_calls:
        try:
            finish_reason = "stop"
            final_chunk_text = ""
            # 'client' here is client_to_use from chat_completions
            async for chunk in client.send_message(bot=bot_name, message=response_content["message"], file_path=image_urls, chatId=chat_id_to_use):
                final_chunk_text = chunk["text"]
                if max_tokens and await helpers.__tokenize(final_chunk_text) >= max_tokens:
                    await client.cancel_message(chunk)
                    finish_reason = "length"
                    break
                pass
            chunk_text_for_tokens = final_chunk_text
        except Exception as e:
            raise HTTPException(detail={"error": {"message": f"Failed to generate completion. Error: {e}", "type": "error", "param": None, "code": 500}}, status_code=500) from e
        
        # --- BEGIN STORE CLIENT & CHAT_ID ON CACHE MISS ---
        if not chat_was_reused and cache_key_to_store and bot_name:
            retrieved_chat_id = client.last_chat_id
            if retrieved_chat_id and client_for_cache_storage is not None and subscription_for_cache_storage is not None:
                app.state.prompt_cache_manager.store_in_cache(
                    cache_key=cache_key_to_store,
                    chat_id=retrieved_chat_id,
                    client_instance=client_for_cache_storage, # Store the client that handled this request
                    subscription_status=subscription_for_cache_storage, # Store its subscription status
                    bot_name=bot_name,
                    source="non_stream"
                )
                # As above, assume last_chat_id is read-only after send
            elif CACHE_LOGGING_ENABLED:
                 app.state.prompt_cache_manager._log(f"[NON_STREAMING_CACHE_STORE_SKIP:NON_STREAM] Could not store. Missing retrieved_chat_id ({retrieved_chat_id is not None}), client_for_cache_storage ({client_for_cache_storage is not None}), or subscription_for_cache_storage ({subscription_for_cache_storage is not None}).")
        # --- END STORE CLIENT & CHAT_ID ON CACHE MISS ---
        
        completion_tokens = await helpers.__tokenize(chunk_text_for_tokens)
        
    else:
        completion_tokens = await helpers.__tokenize(''.join([str(tool_call["name"]) + str(tool_call["arguments"]) for tool_call in raw_tool_calls]))
        # For OpenAI compatibility, 'chunk' would be what the API uses internally.
        # We provide chunk["text"] for content, so if it's raw_tool_calls, there's no direct "text" for content.
        # The MessageResponse in choices handles this with `content=None if raw_tool_calls else chunk["text"]`
        # We assign an empty string for chunk_text_for_tokens to ensure it exists.
        chunk_text_for_tokens = "" # Represents the final message content, empty for tool_calls
        finish_reason = "tool_calls"
        
    content = ChatCompletionResponse(
        id=f"chatcmpl-{completion_id}",
        object="chat.completion",
        created=await helpers.__generate_timestamp(),
        model=model,
        usage=ChatCompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=MessageResponse(role="assistant", content=None if raw_tool_calls else chunk_text_for_tokens,
                                        tool_calls=[ChatCompletionMessageToolCall(
                                            id=f"call-{await helpers.__generate_completion_id()}",
                                            function=FunctionCall(name=tool_call["name"], arguments=orjson.dumps(tool_call["arguments"]))) for tool_call in raw_tool_calls] if raw_tool_calls else None),
                finish_reason=finish_reason,
            )
        ],  
    )
    
    return ORJSONResponse(content.model_dump())


async def rotate_token(tokens) -> Tuple[AsyncPoeApi, bool]:
    if len(tokens) == 0:
        raise HTTPException(detail={"error": {"message": "All tokens have been used. Please add more tokens.", "type": "error", "param": None, "code": 402}}, status_code=402)
    token = random.choice(tokens)
    client = await AsyncPoeApi(token).create()
    settings = await client.get_settings()
    if settings["messagePointInfo"]["messagePointBalance"] <= 20:
        tokens.remove(token)
        return await rotate_token(tokens)
    subscriptions = settings["subscription"]["isActive"]
    return client, subscriptions


if __name__ == "__main__":
    CommandLineInterface().run(["api:app", "--bind", "127.0.0.1", "--port", "8000"])
    
    
def start_server(tokens: list, address: str="127.0.0.1", port: str="8000"):
    if not isinstance(tokens, list):
        raise TypeError("Tokens must be a list.")
    if not all(isinstance(token, dict) for token in tokens):
        raise TypeError("Tokens must be a list of dictionaries.")
    app.state.tokens = tokens
    CommandLineInterface().run(["poe_api_wrapper.openai.api:app", "--bind", f"{address}", "--port", f"{port}"])