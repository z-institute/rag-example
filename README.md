# pdf_rag

## Installation

Python version recommendation: 3.11

Install the LangChain CLI if you haven't yet

```bash
pip install -U langchain-cli
```

```bash
poetry install
```

## Usage of Website loading

```bash
python importer\save_notion_html.py
```

then

```bash
python importer\html_process.py
python3 importer/load_and_process.py

poetry run python importer/load_and_process.py
poetry run python app/server.py
```

## Usage of Telegram bot

```bash
poetry run python bot.py
```

## Adding packages

```bash
poetry add package_name
```

## Setup LangSmith (Optional)

LangSmith will help us trace, monitor and debug LangChain applications.
LangSmith is currently in private beta, you can sign up [here](https://smith.langchain.com/).
If you don't have access, you can skip this section

```shell
export langchain_tracing_v2=true
export LANGCHAIN_API_KEY=<your-api-key>
export LANGCHAIN_PROJECT=<your-project>  # if not specified, defaults to "default"
```

## Launch LangServe

```bash
langchain serve
```

## Running in Docker

This project folder includes a Dockerfile that allows you to easily build and host your LangServe app.

### Building the Image

To build the image, you simply:

```shell
docker build . -t my-langserve-app
```

If you tag your image with something other than `my-langserve-app`,
note it for use in the next step.

### Running the Image Locally

To run the image, you'll need to include any environment variables
necessary for your application.

In the below example, we inject the `OPENAI_API_KEY` environment
variable with the value set in my local environment
(`$OPENAI_API_KEY`)

We also expose port 8080 with the `-p 8080:8080` option.

```shell
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -p 8080:8080 my-langserve-app
```
