import os

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import MongoDBVectorSearchConfig, MongoDBVectorSearchTool


@CrewBase
class AnswerCrewaiPromptCrew:
    agents: list[BaseAgent]
    tasks: list[Task]

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def linguist(self) -> Agent:
        return Agent(
            config=self.agents_config["linguist"],
        )

    @agent
    def crewai_expert(self) -> Agent:
        return Agent(
            config=self.agents_config["crewai_expert"],
            tools=[
                MongoDBVectorSearchTool(
                    connection_string=os.getenv("MONGODB_CONNECTION_STRING"),
                    database_name=os.getenv("MONGODB_DATABASE_NAME"),
                    collection_name=os.getenv("MONGODB_COLLECTION_NAME"),
                    dimensions=3072,
                    query_config=MongoDBVectorSearchConfig(limit=10),
                )
            ],
        )

    @task
    def identify_language(self) -> Task:
        return Task(
            config=self.tasks_config["identify_language"],
        )

    @task
    def research_proper_answer(self) -> Task:
        return Task(
            config=self.tasks_config["research_proper_answer"],
        )

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
