from mcp.server.fastmcp import Context, FastMCP

from src.crewai_expert.main import CrewaiExpertFlow

mcp = FastMCP("CrewAI Expert MCP Server")


@mcp.tool()
async def crewai_question(prompt: str, ctx: Context) -> str:
    output = await CrewaiExpertFlow(context=ctx).kickoff_async(
        inputs={"prompt": prompt}
    )

    return output["final_answer"]


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
