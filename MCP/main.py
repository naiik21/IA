import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from mcp.server.fastmcp import FastMCP, Context
import os
from dotenv import load_dotenv
import discord
from discord.ext import commands
import jsons

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_BOT_TOKEN")

intents = discord.Intents.default()
intents.members = True
intents.presences = True
intents.message_content = True


@dataclass
class AppContext:
    bot: discord.Client

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    bot = discord.Client(intents=intents)
    await bot.login(DISCORD_TOKEN)
    asyncio.create_task(bot.connect())
    try:
        yield AppContext(bot=bot)
    finally:
        bot.close()

mcp = FastMCP("Test", lifespan=app_lifespan)

@mcp.tool()
def getOS() -> str:
    return os.uname().sysname

async def _getBot(ctx: Context) -> discord.Client:
    bot: discord.Client = ctx.request_context.lifespan_context.bot
    if not bot.is_ready():
        await bot.wait_until_ready()
    return bot

async def _getGuild(bot: discord.Client, guildId: str, ctx: Context) -> discord.Guild | None:
    guild = None
    gId = int(guildId, 16)
    for g in bot.guilds:
        if g.id == gId:
            guild = g
    return guild

@mcp.tool()
async def getDiscordGuilds(ctx: Context) -> str:
    bot = await _getBot(ctx)
    guilds = [{
        "guildId": hex(guild.id), 
        "name": guild.name
    } for guild in bot.guilds]
    return jsons.dump(guilds)

@mcp.tool()
async def getDiscordChannelsInGuild(ctx: Context, guildId: str) -> str:
    bot = await _getBot(ctx)
    guild = await _getGuild(bot, guildId, ctx)
    channels = [{
        "channelId": hex(channel.id),
        "name": channel.name,
        "type": str(channel.type)
    } for channel in guild.channels]
    return jsons.dump(channels)

@mcp.tool()
async def getDiscordUsersInGuild(ctx: Context, guildId: str, showBots: bool = False) -> str:
    bot = await _getBot(ctx)

    members = []
    guild = await _getGuild(bot, guildId, ctx)
    async for member in guild.fetch_members(limit=None):
        member2 = guild.get_member(member.id)
        members.append({
            "memberId": hex(member2.id),
            "name": member2.name,
            "display_name": member2.display_name,
            "status": str(member2.status),
            "bot": member2.bot
        })
    if not showBots:
        return jsons.dump([m for m in members if not m["bot"]])
    return jsons.dump(members)


if __name__ == "__main__":
    mcp.run(transport="stdio")