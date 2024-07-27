import os
import re
import aiohttp
import discord
from discord.ext import commands, tasks
from dotenv import load_dotenv

from PIL import Image
import io
import base64

import vertexai
import httpx
from google.auth import default
from google.auth.transport.requests import Request

# Load environment variables
load_dotenv()
DISCORD_BOT_TOKEN = os.getenv("DISCORD_BOT_TOKEN")
GCP_REGION = os.getenv("GCP_REGION")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")
TEMPERATURE = 0.5
TOP_P = 0.95
MAX_TOKEN = 2048
vertexai.init(project=GCP_PROJECT_ID, location=GCP_REGION, )

# Obtaining and Updating Google Cloud Credentials
credentials, _ = default()
auth_request = Request()
credentials.refresh(auth_request)


def build_endpoint_url(
    region: str,
    project_id: str,
    model_name: str,
    model_version: str,
    streaming: bool = False,
):
    base_url = f"https://{GCP_REGION}-aiplatform.googleapis.com/v1/"
    project_fragment = f"projects/{GCP_PROJECT_ID}"
    location_fragment = f"locations/{GCP_REGION}"
    specifier = "streamRawPredict" if streaming else "rawPredict"
    model_fragment = f"publishers/mistralai/models/{model_name}@{model_version}"
    url = f"{base_url}{'/'.join([project_fragment, location_fragment, model_fragment])}:{specifier}"
    return url

model = "mistral-large"
model_version = "2407"
LLM_MODEL = f"{model}@{model_version}"

# Build URL
url = build_endpoint_url(
    region=GCP_REGION,
    project_id=GCP_PROJECT_ID,
    model_name=model,
    model_version=model_version,
    streaming=False
)


# Define query headers
headers = {}


# The history is counted by the number of pairs of exchanges between the user and the assistant.
MAX_HISTORY = 2*int(os.getenv("MAX_HISTORY", "0"))  # Default to 0 if not set

# The maximum number of characters per Discord message
MAX_DISCORD_LENGTH = 2000
# 
SYSTEM_MESSAGE = {
    "role": "system",
    "content":'''
You are a helpful AI assistant, working on Discord servers.
'''
}



# Initialize Discord bot
bot = commands.Bot(command_prefix="!", intents=discord.Intents.default())

message_history = {}


@tasks.loop(minutes=55)
async def refresh_credentials():
    global credentials, headers
    credentials.refresh(auth_request)
    headers = {
        "Authorization": f"Bearer {credentials.token}",
        "Accept": "application/json",
    }
    # client.api_key = credentials.token
    print("Credentials refreshed")

@bot.event
async def on_ready():
    refresh_credentials.start()
    print(f"{LLM_MODEL} Bot Logged in as {bot.user}")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.mention_everyone:
        await message.channel.send(f'{bot.user}です')
        return

    if bot.user.mentioned_in(message) or isinstance(message.channel, discord.DMChannel):
        cleaned_text = clean_discord_message(message.content)
        async with message.channel.typing():
            if message.attachments:
                await process_attachments(message, cleaned_text)
            else:
                await process_text_message(message, cleaned_text)

async def process_attachments(message, cleaned_text):
    # print(f"New Image Message FROM: {message.author.id}: {cleaned_text}")
    for attachment in message.attachments:
        file_extension = os.path.splitext(attachment.filename.lower())[1]
        # ext_to_mime = {'.png': "image/png", '.jpg': "image/jpeg", '.jpeg': "image/jpeg", '.gif': "image/gif", '.webp': "image/webp"}
        ext_to_mime = {
            # '.png': "image/png", 
            # '.jpg': "image/jpeg", 
            # '.jpeg': "image/jpeg", 
            # '.gif': "image/gif", 
            # '.webp': "image/webp",
            '.txt': "text/plain",
            '.md': "text/markdown",
            '.csv': "text/csv",
            '.json': "application/json",
            '.xml': "application/xml",
            '.html': "text/html",
            '.ini': "text/plain",
            '.log': "text/plain",
            '.yaml': "text/yaml",
            '.yml': "text/yaml",
            '.c': "text/x-c",
            '.h': "text/x-c",
            '.cpp': "text/x-c++",
            '.hpp': "text/x-c++",
            '.py': "text/x-python",
            '.rs': "text/x-rust",
            '.js': "text/javascript",
            '.cs': "text/x-csharp",
            '.php': "text/x-php",
            '.rb': "text/x-ruby",
            '.pl': "text/x-perl",
            '.pm': "text/x-perl",
            '.swift': "text/x-swift",
            '.r': "text/x-r",
            '.R': "text/x-r",
            '.go': "text/x-go"
        }
        if file_extension in ext_to_mime:
            if file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.webp']:
                await message.add_reaction('🎨')
                mime_type = ext_to_mime[file_extension]
                async with aiohttp.ClientSession() as session:
                    async with session.get(attachment.url) as resp:
                        if resp.status != 200:
                            await message.channel.send('Unable to download the image.')
                            return
                        image_data = await resp.read()
                        resized_image_stream = resize_image_if_needed(image_data, file_extension)
                        resized_image_data = resized_image_stream.getvalue()
                        encoded_image_data = base64.b64encode(resized_image_data).decode("utf-8")
                        response_text = await generate_response_with_image_and_text(encoded_image_data, cleaned_text, mime_type)
                        await split_and_send_messages(message, response_text, MAX_DISCORD_LENGTH)
                        return
            else:
                await message.add_reaction('📄')
                async with aiohttp.ClientSession() as session:
                    async with session.get(attachment.url) as resp:
                        if resp.status != 200:
                            await message.channel.send('Unable to download the text file.')
                            return
                        text_data = await resp.text()
                        # cleaned_textが空でない場合のみテキストに追加
                        combined_text = f"{cleaned_text}\n{text_data}" if cleaned_text else text_data
                        await process_text_message(message, combined_text)
                        return
        else:
            supported_extensions = ', '.join(ext_to_mime.keys())
            await message.channel.send(f"🗑️ Unsupported file extension. Supported extensions are: {supported_extensions}")


async def process_text_message(message, cleaned_text):
    # print(f"New Message FROM: {message.author.id}: {cleaned_text}")
    # Use a regex to find 'RESET' as a whole word, case-insensitively
    # Match only if the message is exactly "RESET"
    if re.search(r'^RESET$', cleaned_text, re.IGNORECASE):
        message_history.pop(message.author.id, None)
        await message.channel.send(f"🧹 History Reset for user: {message.author.name}")
        return

    await message.add_reaction('💬')
    update_message_history(message.author.id, cleaned_text, "user")
    formatted_history = get_formatted_message_history(message.author.id)
    response_text = await generate_response_with_text(formatted_history)
    update_message_history(message.author.id, response_text, "assistant")
    await split_and_send_messages(message, response_text, MAX_DISCORD_LENGTH)

# def preprocess_message(content):
#     if content.startswith("assistant"):
#         return content[len("assistant"):]
#     return content

async def generate_response_with_text(message_text):
    data = {
        "model": model,
        "messages": message_text,
        "stream": False,
    }
    try:
        response = None
        async with httpx.AsyncClient() as client:
            resp = await client.post(url, json=data, headers=headers, timeout=None)
            response_json = resp.json()
            response = response_json['choices'][0]['message']['content']
        return response
        # processed_response = preprocess_message(response)
        # return processed_response
    except httpx.RequestError as exc:
        return f"An error occurred while requesting {exc.request.url!r}."


async def generate_response_with_image_and_text(image_data, text, mime_type):
    # although Llama cannot handle multimodal data
    # try:
    #     answer = await client.chat.completions.create(
    #         model=LLM_MODEL,
    #         max_tokens=MAX_TOKEN,
    #         messages=[{
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": text if text else 'What is this a picture of?'},
    #                 {"type": "image", "source": {"type": "base64", "media_type": mime_type, "data": image_data}}
    #             ]
    #         }]
    #     )
    #     return answer.choices[0].message.content
    # except Exception as e:
    #     return f"Error: Failed to generate response due to {str(e)}"
    
    return f"Not Implemented yet"



def update_message_history(user_id, text, message_type):
    global message_history
    # prefixed_message = f"{message_type}: {text}"
    # Construct the new message as a dictionary
    new_message = {'role': message_type, 'content': [text]}
    if user_id in message_history:
        message_history[user_id].append(new_message)
        if message_type == 'assistant' and len(message_history[user_id]) > MAX_HISTORY:
            message_history[user_id].pop(0)
            if len(message_history[user_id]) > 0:
                message_history[user_id].pop(0)
    else:
        message_history[user_id] = [new_message]

def get_formatted_message_history(user_id):
    # Check if the user has any messages
    if user_id not in message_history or not message_history[user_id]:
        return "No messages found for this user."
    
    # Format each message in the history
    formatted_messages = []
    # System prompt
    formatted_messages.append(SYSTEM_MESSAGE)

    for message in message_history[user_id]:
        role = message['role']
        content = " ".join(message['content'])
        formatted_messages.append({"role": role, "content": content})
    
    return formatted_messages

def clean_discord_message(input_string):
    bracket_pattern = re.compile(r'<[^>]+>')
    return bracket_pattern.sub('', input_string)

def resize_image_if_needed(image_bytes, file_extension, max_size_mb=1, step=10):
    format_map = {'.png': 'PNG', '.jpg': 'JPEG', '.jpeg': 'JPEG', '.gif': 'GIF', '.webp': 'WEBP'}
    img_format = format_map.get(file_extension.lower(), 'JPEG')
    img_stream = io.BytesIO(image_bytes)
    img = Image.open(img_stream)
    while img_stream.getbuffer().nbytes > max_size_mb * 1024 * 1024:
        width, height = img.size
        img = img.resize((int(width * (100 - step) / 100), int(height * (100 - step) / 100)), Image.ANTIALIAS)
        img_stream = io.BytesIO()
        img.save(img_stream, format=img_format)
    return img_stream

async def split_and_send_messages(message_system, text, max_length):
    """
    Splits the given text into chunks that respect word boundaries and sends them
    using the provided message system. Chunks are up to max_length characters long.

    :param message_system: An object representing the Discord messaging system,
                           assumed to have a `channel.send` method for sending messages.
    :param text: The text to be sent.
    :param max_length: The maximum length of each message chunk.
    """
    start = 0
    while start < len(text):
        # If remaining text is within the max_length, send it as one chunk.
        if len(text) - start <= max_length:
            await message_system.channel.send(text[start:])
            break

        # Find the last whitespace character before the max_length limit.
        end = start + max_length
        while end > start and text[end-1] not in ' \n\r\t':
            end -= 1

        # If no suitable whitespace is found, force break at max_length.
        if end == start:
            end = start + max_length

        # Send the text from start to end.
        await message_system.channel.send(text[start:end].strip())
        
        # Update start position for next iteration to continue after the last whitespace.
        start = end

# Run the bot
bot.run(DISCORD_BOT_TOKEN)
