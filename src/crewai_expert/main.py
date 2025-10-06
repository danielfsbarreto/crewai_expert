from crewai.flow import Flow, listen, router, start
from mcp.server.fastmcp import Context

from src.crewai_expert.crews import AnswerCrewaiPromptCrew
from src.crewai_expert.services import DocFilesChunkingService
from src.crewai_expert.types import CrewaiExpertState


class CrewaiExpertFlow(Flow[CrewaiExpertState]):
    def __init__(self, context: Context, *args, **kwargs):
        self.context = context
        self.doc_files_chunking_service = DocFilesChunkingService()

        super().__init__(*args, **kwargs)

    @start()
    async def validate_inputs(self):
        await self.context.report_progress(
            progress=0, total=3, message="Validating inputs..."
        )
        if self.state.run_type == "answer_prompt" and self.state.prompt is None:
            raise ValueError("The input `prompt` needs to be provided")

    @router(validate_inputs)
    async def identify_path(self):
        await self.context.report_progress(
            progress=1, total=3, message="Identifying path..."
        )
        return self.state.run_type

    @listen("update_embeddings")
    async def update_embeddings_path(self):
        await self.context.report_progress(
            progress=2, total=3, message="Updating embeddings..."
        )
        await self.doc_files_chunking_service.call()

    @listen("answer_prompt")
    async def come_up_with_curated_answer(self):
        await self.context.report_progress(
            progress=2, total=3, message="Coming up with a curated answer..."
        )
        result = await (
            AnswerCrewaiPromptCrew(
                collection_name=self.doc_files_chunking_service.collection_name
            )
            .crew()
            .kickoff_async(inputs={"prompt": self.state.prompt})
        )
        self.state.final_answer = result.raw

        return {"final_answer": self.state.final_answer}
