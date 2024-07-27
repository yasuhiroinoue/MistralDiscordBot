
# MistralDiscordBot

MistralDiscordBot is a Discord bot designed to interact with users in a Discord server using a language model deployed on Google Cloud Platform's AI services. The bot can respond to user queries and provide assistance using natural language processing.

## Features

- Responds to user queries in a Discord server.
- Utilizes a language model from Google Cloud AI services.
- Supports configurable parameters such as temperature, top-p, and max tokens.
- Keeps a history of exchanges for context-aware responses.
- Uses environment variables for configuration.

## Prerequisites

- Python 3.7+
- Discord API token
- Google Cloud Platform project with AI services enabled
- Required Python libraries (see `requirements.txt`)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/MistralDiscordBot.git
   cd MistralDiscordBot
   ```

2. Create a virtual environment and activate it:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:

   Create a `.env` file in the project root directory and add the following variables:

   ```env
   DISCORD_BOT_TOKEN=your_discord_bot_token
   GCP_REGION=your_gcp_region
   GCP_PROJECT_ID=your_gcp_project_id
   MAX_HISTORY=2  # Or any desired value
   ```

## Usage

1. Run the bot:

   ```bash
   python MistralDiscordBot.py
   ```

2. Invite the bot to your Discord server using the OAuth2 URL from the Discord Developer Portal.

3. Interact with the bot in your server by mentioning it or using specific commands.

## Configuration

- **Temperature:** Controls the randomness of the model's responses. Higher values (e.g., 0.8) make output more random, while lower values (e.g., 0.2) make it more focused and deterministic.
- **Top-p:** Controls the cumulative probability of token selection. Used for nucleus sampling.
- **Max Tokens:** Limits the maximum number of tokens in the generated response.
- **Max History:** Defines the maximum number of exchanges to keep in the conversation history.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [discord.py](https://github.com/Rapptz/discord.py) - Python wrapper for the Discord API.
- [Google Cloud AI Platform](https://cloud.google.com/ai-platform) - Platform for training and deploying machine learning models.
- [dotenv](https://github.com/theskumar/python-dotenv) - Python library for reading environment variables from a `.env` file.
