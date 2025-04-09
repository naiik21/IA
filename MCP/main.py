from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv
import discord
from discord.ext import commands

load_dotenv()

DISCORD_TOKEN=os.getenv('DISCORD_BOT_TOKEN')

intents = discord.Intents.default()
intents.members=True
intents.presences= True

bot_ready=False

bot= commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    bot_ready=True

mcp= FastMCP("Test")

@mcp.tool()
def getOS()-> str:
    return os.uname().sysname

@mcp.tool()
async def getDiscordUsers()-> list[discord.Member]:
    members= list()
    for guild in bot.guilds:
        async for member in guild.fetch_members(limit=None):
            members.insert(member)
    return members

if __name__ == "__main__":
    mcp.run(transport="stdio")



# def main():
#     print("Hello from mcp!")


# if __name__ == "__main__":
#     main()
