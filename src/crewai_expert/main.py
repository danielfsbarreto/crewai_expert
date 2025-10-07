from crewai.flow import Flow, listen, router, start

from src.crewai_expert.crews import AnswerCrewaiPromptCrew
from src.crewai_expert.services import DocFilesChunkingService
from src.crewai_expert.types import CrewaiExpertState


class CrewaiExpertFlow(Flow[CrewaiExpertState]):
    def __init__(self, *args, **kwargs):
        self.doc_files_chunking_service = DocFilesChunkingService()

        super().__init__(*args, **kwargs)

    @start()
    def validate_inputs(self):
        if self.state.run_type == "answer_prompt" and self.state.prompt is None:
            raise ValueError("The input `prompt` needs to be provided")

    @router(validate_inputs)
    def identify_path(self):
        return self.state.run_type

    @listen("update_embeddings")
    async def update_embeddings_path(self):
        await self.doc_files_chunking_service.call()

    @listen("answer_prompt")
    def come_up_with_curated_answer(self):
        result = (
            AnswerCrewaiPromptCrew(
                collection_name=self.doc_files_chunking_service.collection_name
            )
            .crew()
            .kickoff(inputs={"prompt": self.state.prompt})
        )
        self.state.final_answer = result.raw

        return {"final_answer": self.state.final_answer}


def kickoff():
    CrewaiExpertFlow().kickoff(
        inputs={"prompt": "When to use a Flow instead of a Crew?"}
    )


if __name__ == "__main__":
    kickoff()
