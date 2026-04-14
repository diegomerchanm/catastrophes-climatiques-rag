"""Test minimal Chainlit."""
import chainlit as cl


@cl.on_chat_start
async def start():
    await cl.Message(content="Test OK").send()


@cl.on_message
async def main(message: cl.Message):
    await cl.Message(content=f"Tu as dit : {message.content}").send()
