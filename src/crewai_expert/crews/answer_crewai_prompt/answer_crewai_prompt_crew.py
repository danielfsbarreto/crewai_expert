import os
from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import QdrantVectorSearchTool


@CrewBase
class AnswerCrewaiPromptCrew:
    agents: List[BaseAgent]
    tasks: List[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    @agent
    def linguist(self) -> Agent:
        return Agent(
            config=self.agents_config["linguist"],  # type: ignore[index]
        )

    @agent
    def crewai_expert(self) -> Agent:
        return Agent(
            config=self.agents_config["crewai_expert"],  # type: ignore[index]
            tools=[
                QdrantVectorSearchTool(
                    qdrant_config={
                        "qdrant_url": os.getenv("QDRANT_URL"),
                        "qdrant_api_key": os.getenv("QDRANT_API_KEY"),
                        "collection_name": self.collection_name,
                        "limit": 5,
                    }
                )
            ],
        )

    @task
    def identify_language(self) -> Task:
        return Task(
            config=self.tasks_config["identify_language"],  # type: ignore[index]
        )

    @task
    def research_proper_answer(self) -> Task:
        return Task(
            config=self.tasks_config["research_proper_answer"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
