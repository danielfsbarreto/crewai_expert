from crewai.flow import Flow, listen, router, start

from crewai_expert.crews import AnswerCrewaiPromptCrew
from crewai_expert.services import DocFilesChunkingService
from crewai_expert.types import CrewaiExpertState


class CrewaiExpertFlow(Flow[CrewaiExpertState]):
    doc_files_chunking_service = DocFilesChunkingService()

    @start()
    def validate_inputs(self):
        if self.state.run_type == "answer_prompt" and self.state.prompt is None:
            raise ValueError("The input `prompt` needs to be provided")

    @router(validate_inputs)
    def identify_path(self):
        return self.state.run_type

    @listen("update_embeddings")
    async def update_embeddings_path(self):
        embeddings_saved = await self.doc_files_chunking_service.call()

        return {"embeddings_saved": embeddings_saved}

    @listen("answer_prompt")
    def come_up_with_curated_answer(self):
        result = (
            AnswerCrewaiPromptCrew()
            .crew()
            .kickoff(inputs={"prompt": self.state.prompt})
        )
        self.state.final_answer = result.raw

        return {"final_answer": self.state.final_answer}


def kickoff():
    CrewaiExpertFlow().kickoff(
        inputs={"run_type": "update_embeddings"}
        # inputs={"prompt": "How to build custom tools in CrewAI?"}
        # inputs={"prompt": "How to decide when to build a crew vs flows?"}
    )


if __name__ == "__main__":
    kickoff()
