"""Multi-Agent Coordination module for OS Multi-Science.

Provides message-passing agents, DAG-based work plans, weighted consensus
voting, and a coordinator that orchestrates the full pipeline.

All communication happens via explicit Message objects -- no shared mutable
state between agents.  The coordinator maintains a full audit trail of every
message routed through the system.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AgentRole(Enum):
    PROFILER = "profiler"       # Analyzes system characteristics
    ROUTER = "router"           # Selects method kit
    EXECUTOR = "executor"       # Runs a specific model
    EVALUATOR = "evaluator"     # Computes ICM, CRC, anti-spurious
    SENTINEL = "sentinel"       # Monitors for early warnings & anomalies


class MessageType(Enum):
    TASK = "task"               # Assign work
    RESULT = "result"           # Return results
    VOTE = "vote"               # Cast a vote in consensus
    ALERT = "alert"             # Early warning or anomaly
    STATUS = "status"           # Status update


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Message:
    """An immutable message passed between agents."""
    id: str
    sender: str
    receiver: str
    msg_type: MessageType
    content: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class AgentState:
    """Mutable internal state of a single agent."""
    status: str = "idle"          # idle | working | done | error
    current_task: str | None = None
    results: list[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """Base agent in the multi-agent system.

    Each agent has a *role*, a list of *capabilities* (free-form tags), an
    internal *state*, and an *inbox* of unprocessed messages.  An optional
    *handler* callback is invoked for each incoming message.
    """

    def __init__(
        self,
        name: str,
        role: AgentRole,
        capabilities: list[str] | None = None,
    ):
        self.name = name
        self.role = role
        self.capabilities = capabilities or []
        self.state = AgentState()
        self.inbox: list[Message] = []
        self._handler: Callable[[Message], Any] | None = None

    # -- handler ------------------------------------------------------------

    def set_handler(self, handler: Callable[[Message], Any]) -> None:
        """Set the message handler function."""
        self._handler = handler

    # -- messaging ----------------------------------------------------------

    def receive_message(self, message: Message) -> None:
        """Receive and optionally process a message.

        The message is always appended to the inbox.  If a handler is
        registered, it is called immediately; otherwise the message stays
        pending until :meth:`process_inbox` is called.
        """
        self.inbox.append(message)
        if self._handler is not None:
            self._handler(message)

    def process_inbox(self) -> list[Message]:
        """Process all pending messages and return response messages.

        If no handler is set, messages are simply marked as processed and
        an empty list is returned.  If a handler *is* set, it is called for
        every message in the inbox.  Any non-``None`` return value from the
        handler that is a :class:`Message` (or dict) is collected and
        returned.
        """
        responses: list[Message] = []
        pending = list(self.inbox)
        self.inbox.clear()

        for msg in pending:
            if self._handler is not None:
                result = self._handler(msg)
                if isinstance(result, Message):
                    responses.append(result)
            # Store content in results for audit
            self.state.results.append(msg.content)

        return responses

    def create_message(
        self,
        receiver: str,
        msg_type: MessageType,
        content: Any,
    ) -> Message:
        """Create a new message originating from this agent."""
        return Message(
            id=str(uuid.uuid4()),
            sender=self.name,
            receiver=receiver,
            msg_type=msg_type,
            content=content,
        )

    # -- repr ---------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Agent(name={self.name!r}, role={self.role.value!r}, "
            f"status={self.state.status!r})"
        )


# ---------------------------------------------------------------------------
# WorkPlan
# ---------------------------------------------------------------------------

class WorkPlan:
    """A plan of agent-task assignments with DAG-like dependencies.

    Each *step* is a dict with keys ``agent_name``, ``task``, and
    ``depends_on`` (a list of task names that must complete first).
    """

    def __init__(self) -> None:
        self.steps: list[dict] = []  # [{agent_name, task, depends_on}]

    def add_step(
        self,
        agent_name: str,
        task: str,
        depends_on: list[str] | None = None,
    ) -> None:
        """Add a step to the plan."""
        self.steps.append({
            "agent_name": agent_name,
            "task": task,
            "depends_on": depends_on or [],
        })

    def get_ready_steps(self, completed: set[str]) -> list[dict]:
        """Return steps whose dependencies have all been satisfied.

        Only steps whose *task* is not yet in *completed* and whose
        ``depends_on`` entries are all in *completed* are returned.
        """
        ready: list[dict] = []
        for step in self.steps:
            if step["task"] in completed:
                continue
            if all(dep in completed for dep in step["depends_on"]):
                ready.append(step)
        return ready

    def is_complete(self, completed: set[str]) -> bool:
        """Return ``True`` when every step's task is in *completed*."""
        return all(step["task"] in completed for step in self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        return f"WorkPlan(steps={len(self.steps)})"


# ---------------------------------------------------------------------------
# ConsensusProtocol
# ---------------------------------------------------------------------------

class ConsensusProtocol:
    """Weighted voting protocol for agent decisions.

    Each agent may cast a single vote (any hashable value).  Tallying
    aggregates votes by weight, and :meth:`get_winner` returns the
    option with the highest cumulative weight.
    """

    def __init__(
        self,
        agents: list[Agent],
        weights: dict[str, float] | None = None,
    ):
        self.agents: dict[str, Agent] = {a.name: a for a in agents}
        self.weights: dict[str, float] = weights or {
            a.name: 1.0 for a in agents
        }
        self.votes: dict[str, Any] = {}

    # -- voting -------------------------------------------------------------

    def cast_vote(self, agent_name: str, vote: Any) -> None:
        """Record *agent_name*'s vote.

        Raises ``KeyError`` if the agent is not registered.
        """
        if agent_name not in self.agents:
            raise KeyError(f"Unknown agent: {agent_name!r}")
        self.votes[agent_name] = vote

    def tally(self) -> dict[str, float]:
        """Aggregate votes into ``{option: weighted_total}``."""
        totals: dict[str, float] = {}
        for agent_name, vote in self.votes.items():
            w = self.weights.get(agent_name, 1.0)
            key = str(vote)
            totals[key] = totals.get(key, 0.0) + w
        return totals

    def get_winner(self) -> Any:
        """Return the option with the highest cumulative weight.

        Raises ``ValueError`` when no votes have been cast.
        """
        totals = self.tally()
        if not totals:
            raise ValueError("No votes have been cast")
        return max(totals, key=lambda k: totals[k])

    def is_unanimous(self) -> bool:
        """Return ``True`` if every registered agent voted the same way."""
        if not self.votes:
            return False
        unique = set(str(v) for v in self.votes.values())
        # Must also have a vote from every registered agent
        if set(self.votes.keys()) != set(self.agents.keys()):
            return False
        return len(unique) == 1

    def reset(self) -> None:
        """Clear all votes (keeps agents and weights)."""
        self.votes.clear()

    def __repr__(self) -> str:
        return (
            f"ConsensusProtocol(agents={len(self.agents)}, "
            f"votes={len(self.votes)})"
        )


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class Coordinator:
    """Orchestrates multiple agents to solve a problem.

    The coordinator owns agent registration, message routing, work-plan
    execution, and consensus rounds.  Every routed message is appended to
    an immutable audit log.
    """

    def __init__(self, config: Any = None):
        self.agents: dict[str, Agent] = {}
        self.message_log: list[Message] = []
        self.config = config

    # -- agent management ---------------------------------------------------

    def add_agent(self, agent: Agent) -> None:
        """Register an agent.  Overwrites if name already exists."""
        self.agents[agent.name] = agent
        logger.info("Added agent %r (%s)", agent.name, agent.role.value)

    def remove_agent(self, name: str) -> None:
        """Remove an agent by name.  Raises ``KeyError`` if not found."""
        if name not in self.agents:
            raise KeyError(f"Agent not found: {name!r}")
        del self.agents[name]
        logger.info("Removed agent %r", name)

    def get_agent(self, name: str) -> Agent | None:
        """Return the agent with *name*, or ``None``."""
        return self.agents.get(name)

    # -- message routing ----------------------------------------------------

    def send_message(self, message: Message) -> None:
        """Route a message to the target agent and log it."""
        self.message_log.append(message)
        target = self.agents.get(message.receiver)
        if target is not None:
            target.receive_message(message)
            logger.debug(
                "Routed %s -> %s (%s)",
                message.sender,
                message.receiver,
                message.msg_type.value,
            )
        else:
            logger.warning(
                "Message to unknown agent %r (from %s)",
                message.receiver,
                message.sender,
            )

    def broadcast(
        self,
        sender: str,
        msg_type: MessageType,
        content: Any,
    ) -> None:
        """Send a message from *sender* to every registered agent.

        The sender itself is also included as a recipient so that its
        inbox reflects the broadcast.
        """
        for name in self.agents:
            msg = Message(
                id=str(uuid.uuid4()),
                sender=sender,
                receiver=name,
                msg_type=msg_type,
                content=content,
            )
            self.send_message(msg)

    # -- work-plan execution ------------------------------------------------

    def execute_work_plan(self, plan: WorkPlan) -> dict[str, Any]:
        """Execute a work plan, respecting step dependencies.

        Each step is dispatched as a TASK message to the assigned agent.
        The agent's handler (if set) produces a result which is stored.
        Returns ``{task_name: result}`` for every completed step.
        """
        completed: set[str] = set()
        results: dict[str, Any] = {}

        # Iterate until the plan is complete (or no more steps can run).
        max_iterations = len(plan.steps) + 1  # safeguard
        iteration = 0
        while not plan.is_complete(completed) and iteration < max_iterations:
            iteration += 1
            ready = plan.get_ready_steps(completed)
            if not ready:
                # Deadlock -- remaining steps have unsatisfied deps
                logger.warning("Work plan deadlock: no ready steps remain")
                break

            for step in ready:
                agent = self.agents.get(step["agent_name"])
                if agent is None:
                    logger.warning(
                        "Agent %r not found for task %r",
                        step["agent_name"],
                        step["task"],
                    )
                    # Mark complete with None result to avoid infinite loop
                    completed.add(step["task"])
                    results[step["task"]] = None
                    continue

                # Update agent state
                agent.state.status = "working"
                agent.state.current_task = step["task"]

                # Send task message
                task_msg = Message(
                    id=str(uuid.uuid4()),
                    sender="coordinator",
                    receiver=agent.name,
                    msg_type=MessageType.TASK,
                    content=step["task"],
                )
                self.send_message(task_msg)

                # Process the inbox -- collect any response
                responses = agent.process_inbox()
                result_value = None
                if responses:
                    result_value = responses[-1].content
                    for resp in responses:
                        self.message_log.append(resp)
                else:
                    # If the handler did not produce a Message, use the
                    # last item appended to agent.state.results.
                    if agent.state.results:
                        result_value = agent.state.results[-1]

                results[step["task"]] = result_value
                completed.add(step["task"])

                # Update agent state
                agent.state.status = "done"
                agent.state.current_task = None

        return results

    # -- consensus ----------------------------------------------------------

    def run_consensus(
        self,
        topic: str,
        voters: list[str] | None = None,
    ) -> Any:
        """Run a consensus vote among specified agents.

        A VOTE-type task is broadcast to eligible voters.  Each voter's
        handler should return a :class:`Message` whose *content* is the
        vote value.  The weighted winner is returned.

        Parameters
        ----------
        topic:
            A description of the matter being voted on.
        voters:
            Agent names eligible to vote.  Defaults to all agents.
        """
        voter_names = voters or list(self.agents.keys())
        voter_agents = [
            self.agents[n] for n in voter_names if n in self.agents
        ]

        if not voter_agents:
            raise ValueError("No valid voters for consensus")

        protocol = ConsensusProtocol(voter_agents)

        for agent in voter_agents:
            # Send vote request
            vote_msg = Message(
                id=str(uuid.uuid4()),
                sender="coordinator",
                receiver=agent.name,
                msg_type=MessageType.VOTE,
                content=topic,
            )
            self.send_message(vote_msg)

            # Process inbox to get the vote
            responses = agent.process_inbox()
            if responses:
                vote_value = responses[-1].content
            elif agent.state.results:
                vote_value = agent.state.results[-1]
            else:
                vote_value = None

            if vote_value is not None:
                protocol.cast_vote(agent.name, vote_value)

        if not protocol.votes:
            raise ValueError("No votes were cast")

        return protocol.get_winner()

    # -- default agents -----------------------------------------------------

    def create_default_agents(self) -> None:
        """Create a standard set of agents for the OS-MS pipeline.

        Creates:
          - profiler   (PROFILER)
          - router     (ROUTER)
          - executor_1 (EXECUTOR)
          - executor_2 (EXECUTOR)
          - executor_3 (EXECUTOR)
          - evaluator  (EVALUATOR)
          - sentinel   (SENTINEL)
        """
        defaults = [
            Agent("profiler", AgentRole.PROFILER, ["system_analysis", "aesc"]),
            Agent("router", AgentRole.ROUTER, ["method_selection", "catalog"]),
            Agent("executor_1", AgentRole.EXECUTOR, ["model_run"]),
            Agent("executor_2", AgentRole.EXECUTOR, ["model_run"]),
            Agent("executor_3", AgentRole.EXECUTOR, ["model_run"]),
            Agent("evaluator", AgentRole.EVALUATOR, ["icm", "crc", "anti_spurious"]),
            Agent("sentinel", AgentRole.SENTINEL, ["early_warning", "anomaly"]),
        ]
        for agent in defaults:
            self.add_agent(agent)

    # -- introspection ------------------------------------------------------

    def get_message_log(
        self,
        agent_name: str | None = None,
    ) -> list[Message]:
        """Get message log, optionally filtered by agent.

        When *agent_name* is given, only messages where the agent is
        sender **or** receiver are returned.
        """
        if agent_name is None:
            return list(self.message_log)
        return [
            m
            for m in self.message_log
            if m.sender == agent_name or m.receiver == agent_name
        ]

    def summary(self) -> dict[str, Any]:
        """Get coordinator status summary."""
        return {
            "n_agents": len(self.agents),
            "agents": {
                name: {
                    "role": agent.role.value,
                    "status": agent.state.status,
                    "capabilities": agent.capabilities,
                    "inbox_size": len(agent.inbox),
                    "results_count": len(agent.state.results),
                }
                for name, agent in self.agents.items()
            },
            "total_messages": len(self.message_log),
            "messages_by_type": dict(
                Counter(m.msg_type.value for m in self.message_log)
            ),
        }

    def __repr__(self) -> str:
        return (
            f"Coordinator(agents={len(self.agents)}, "
            f"messages={len(self.message_log)})"
        )
