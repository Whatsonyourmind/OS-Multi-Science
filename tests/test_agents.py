"""Comprehensive tests for the multi-agent coordination module.

Covers:
- Agent creation, messaging, inbox processing, handler callbacks
- WorkPlan creation, dependency tracking, step ordering, completeness
- ConsensusProtocol: voting, tallying, weighted voting, unanimity, reset
- Coordinator: agent management, message routing, broadcast
- Work plan execution with dependencies
- Default agent creation
- Message log filtering
- Edge cases and error handling
"""

from __future__ import annotations

import pytest

from agents.coordinator import (
    Agent,
    AgentRole,
    AgentState,
    ConsensusProtocol,
    Coordinator,
    Message,
    MessageType,
    WorkPlan,
)


# ===================================================================
# Agent tests
# ===================================================================

class TestAgent:
    """Tests for the Agent class."""

    def test_creation_defaults(self):
        agent = Agent("alice", AgentRole.PROFILER)
        assert agent.name == "alice"
        assert agent.role == AgentRole.PROFILER
        assert agent.capabilities == []
        assert agent.state.status == "idle"
        assert agent.state.current_task is None
        assert agent.state.results == []
        assert agent.inbox == []

    def test_creation_with_capabilities(self):
        agent = Agent("bob", AgentRole.EXECUTOR, ["model_run", "gpu"])
        assert agent.capabilities == ["model_run", "gpu"]

    def test_receive_message_adds_to_inbox(self):
        agent = Agent("alice", AgentRole.PROFILER)
        msg = Message(
            id="m1", sender="bob", receiver="alice",
            msg_type=MessageType.TASK, content="analyze",
        )
        agent.receive_message(msg)
        assert len(agent.inbox) == 1
        assert agent.inbox[0].id == "m1"

    def test_receive_message_calls_handler(self):
        agent = Agent("alice", AgentRole.PROFILER)
        received = []
        agent.set_handler(lambda m: received.append(m.content))
        msg = Message(
            id="m1", sender="bob", receiver="alice",
            msg_type=MessageType.TASK, content="analyze",
        )
        agent.receive_message(msg)
        assert received == ["analyze"]

    def test_process_inbox_clears_inbox(self):
        agent = Agent("alice", AgentRole.PROFILER)
        msg = Message(
            id="m1", sender="bob", receiver="alice",
            msg_type=MessageType.TASK, content="work",
        )
        agent.receive_message(msg)
        assert len(agent.inbox) == 1
        agent.process_inbox()
        assert len(agent.inbox) == 0

    def test_process_inbox_stores_results(self):
        agent = Agent("alice", AgentRole.PROFILER)
        msg = Message(
            id="m1", sender="bob", receiver="alice",
            msg_type=MessageType.TASK, content={"key": "value"},
        )
        agent.receive_message(msg)
        agent.process_inbox()
        assert {"key": "value"} in agent.state.results

    def test_process_inbox_returns_handler_messages(self):
        agent = Agent("alice", AgentRole.PROFILER)

        def handler(msg: Message) -> Message:
            return agent.create_message(
                msg.sender, MessageType.RESULT, f"done:{msg.content}"
            )

        agent.set_handler(handler)
        msg = Message(
            id="m1", sender="bob", receiver="alice",
            msg_type=MessageType.TASK, content="task_1",
        )
        agent.receive_message(msg)
        responses = agent.process_inbox()
        assert len(responses) == 1
        assert responses[0].sender == "alice"
        assert responses[0].receiver == "bob"
        assert responses[0].content == "done:task_1"

    def test_create_message_fields(self):
        agent = Agent("alice", AgentRole.PROFILER)
        msg = agent.create_message("bob", MessageType.STATUS, "ok")
        assert msg.sender == "alice"
        assert msg.receiver == "bob"
        assert msg.msg_type == MessageType.STATUS
        assert msg.content == "ok"
        assert isinstance(msg.id, str)
        assert len(msg.id) > 0
        assert msg.timestamp > 0

    def test_repr(self):
        agent = Agent("alice", AgentRole.PROFILER)
        r = repr(agent)
        assert "alice" in r
        assert "profiler" in r


# ===================================================================
# WorkPlan tests
# ===================================================================

class TestWorkPlan:
    """Tests for the WorkPlan class."""

    def test_empty_plan(self):
        plan = WorkPlan()
        assert len(plan) == 0
        assert plan.is_complete(set())

    def test_add_step(self):
        plan = WorkPlan()
        plan.add_step("alice", "profile")
        assert len(plan) == 1
        assert plan.steps[0]["agent_name"] == "alice"
        assert plan.steps[0]["task"] == "profile"
        assert plan.steps[0]["depends_on"] == []

    def test_add_step_with_dependencies(self):
        plan = WorkPlan()
        plan.add_step("alice", "profile")
        plan.add_step("bob", "route", depends_on=["profile"])
        assert plan.steps[1]["depends_on"] == ["profile"]

    def test_get_ready_steps_no_deps(self):
        plan = WorkPlan()
        plan.add_step("alice", "t1")
        plan.add_step("bob", "t2")
        ready = plan.get_ready_steps(set())
        assert len(ready) == 2

    def test_get_ready_steps_with_deps(self):
        plan = WorkPlan()
        plan.add_step("alice", "t1")
        plan.add_step("bob", "t2", depends_on=["t1"])
        plan.add_step("charlie", "t3", depends_on=["t1", "t2"])

        # Nothing completed: only t1 is ready
        ready = plan.get_ready_steps(set())
        assert len(ready) == 1
        assert ready[0]["task"] == "t1"

        # t1 completed: t2 becomes ready
        ready = plan.get_ready_steps({"t1"})
        assert len(ready) == 1
        assert ready[0]["task"] == "t2"

        # t1+t2 completed: t3 becomes ready
        ready = plan.get_ready_steps({"t1", "t2"})
        assert len(ready) == 1
        assert ready[0]["task"] == "t3"

    def test_get_ready_steps_skips_completed(self):
        plan = WorkPlan()
        plan.add_step("alice", "t1")
        plan.add_step("bob", "t2")
        ready = plan.get_ready_steps({"t1"})
        assert len(ready) == 1
        assert ready[0]["task"] == "t2"

    def test_is_complete(self):
        plan = WorkPlan()
        plan.add_step("alice", "t1")
        plan.add_step("bob", "t2")
        assert not plan.is_complete(set())
        assert not plan.is_complete({"t1"})
        assert plan.is_complete({"t1", "t2"})

    def test_diamond_dependency(self):
        """Test a diamond-shaped DAG: A -> B, A -> C, B+C -> D."""
        plan = WorkPlan()
        plan.add_step("a1", "A")
        plan.add_step("a2", "B", depends_on=["A"])
        plan.add_step("a3", "C", depends_on=["A"])
        plan.add_step("a4", "D", depends_on=["B", "C"])

        ready = plan.get_ready_steps(set())
        assert [s["task"] for s in ready] == ["A"]

        ready = plan.get_ready_steps({"A"})
        tasks = {s["task"] for s in ready}
        assert tasks == {"B", "C"}

        ready = plan.get_ready_steps({"A", "B"})
        assert [s["task"] for s in ready] == ["C"]

        ready = plan.get_ready_steps({"A", "B", "C"})
        assert [s["task"] for s in ready] == ["D"]


# ===================================================================
# ConsensusProtocol tests
# ===================================================================

class TestConsensusProtocol:
    """Tests for the ConsensusProtocol class."""

    def _make_agents(self, names: list[str]) -> list[Agent]:
        return [Agent(n, AgentRole.EVALUATOR) for n in names]

    def test_cast_and_tally(self):
        agents = self._make_agents(["a", "b", "c"])
        proto = ConsensusProtocol(agents)
        proto.cast_vote("a", "yes")
        proto.cast_vote("b", "yes")
        proto.cast_vote("c", "no")
        totals = proto.tally()
        assert totals["yes"] == 2.0
        assert totals["no"] == 1.0

    def test_get_winner(self):
        agents = self._make_agents(["a", "b", "c"])
        proto = ConsensusProtocol(agents)
        proto.cast_vote("a", "option_A")
        proto.cast_vote("b", "option_A")
        proto.cast_vote("c", "option_B")
        assert proto.get_winner() == "option_A"

    def test_get_winner_no_votes_raises(self):
        agents = self._make_agents(["a"])
        proto = ConsensusProtocol(agents)
        with pytest.raises(ValueError, match="No votes"):
            proto.get_winner()

    def test_weighted_voting(self):
        agents = self._make_agents(["a", "b"])
        weights = {"a": 10.0, "b": 1.0}
        proto = ConsensusProtocol(agents, weights=weights)
        proto.cast_vote("a", "X")
        proto.cast_vote("b", "Y")
        # a's vote has weight 10 vs b's weight 1
        assert proto.get_winner() == "X"

    def test_is_unanimous_true(self):
        agents = self._make_agents(["a", "b", "c"])
        proto = ConsensusProtocol(agents)
        proto.cast_vote("a", "yes")
        proto.cast_vote("b", "yes")
        proto.cast_vote("c", "yes")
        assert proto.is_unanimous()

    def test_is_unanimous_false_different_votes(self):
        agents = self._make_agents(["a", "b"])
        proto = ConsensusProtocol(agents)
        proto.cast_vote("a", "yes")
        proto.cast_vote("b", "no")
        assert not proto.is_unanimous()

    def test_is_unanimous_false_missing_votes(self):
        agents = self._make_agents(["a", "b", "c"])
        proto = ConsensusProtocol(agents)
        proto.cast_vote("a", "yes")
        proto.cast_vote("b", "yes")
        # c has not voted
        assert not proto.is_unanimous()

    def test_is_unanimous_no_votes(self):
        agents = self._make_agents(["a"])
        proto = ConsensusProtocol(agents)
        assert not proto.is_unanimous()

    def test_reset(self):
        agents = self._make_agents(["a", "b"])
        proto = ConsensusProtocol(agents)
        proto.cast_vote("a", "yes")
        proto.cast_vote("b", "no")
        proto.reset()
        assert proto.votes == {}
        # Agents and weights should still be present
        assert len(proto.agents) == 2

    def test_cast_vote_unknown_agent_raises(self):
        agents = self._make_agents(["a"])
        proto = ConsensusProtocol(agents)
        with pytest.raises(KeyError, match="Unknown agent"):
            proto.cast_vote("nonexistent", "yes")

    def test_vote_override(self):
        """Casting a second vote should override the first."""
        agents = self._make_agents(["a"])
        proto = ConsensusProtocol(agents)
        proto.cast_vote("a", "yes")
        proto.cast_vote("a", "no")
        assert proto.votes["a"] == "no"
        assert proto.get_winner() == "no"


# ===================================================================
# Coordinator tests
# ===================================================================

class TestCoordinator:
    """Tests for the Coordinator class."""

    def test_add_and_get_agent(self):
        coord = Coordinator()
        agent = Agent("alice", AgentRole.PROFILER)
        coord.add_agent(agent)
        assert coord.get_agent("alice") is agent

    def test_get_nonexistent_agent(self):
        coord = Coordinator()
        assert coord.get_agent("nobody") is None

    def test_remove_agent(self):
        coord = Coordinator()
        coord.add_agent(Agent("alice", AgentRole.PROFILER))
        coord.remove_agent("alice")
        assert coord.get_agent("alice") is None

    def test_remove_nonexistent_raises(self):
        coord = Coordinator()
        with pytest.raises(KeyError, match="Agent not found"):
            coord.remove_agent("nobody")

    def test_send_message(self):
        coord = Coordinator()
        alice = Agent("alice", AgentRole.PROFILER)
        coord.add_agent(alice)
        msg = Message(
            id="m1", sender="coord", receiver="alice",
            msg_type=MessageType.TASK, content="go",
        )
        coord.send_message(msg)
        assert len(alice.inbox) == 1
        assert len(coord.message_log) == 1

    def test_send_message_unknown_receiver(self):
        """Message to unknown agent should still be logged, not crash."""
        coord = Coordinator()
        msg = Message(
            id="m1", sender="coord", receiver="nobody",
            msg_type=MessageType.TASK, content="go",
        )
        coord.send_message(msg)
        assert len(coord.message_log) == 1

    def test_broadcast(self):
        coord = Coordinator()
        coord.add_agent(Agent("a", AgentRole.EXECUTOR))
        coord.add_agent(Agent("b", AgentRole.EXECUTOR))
        coord.add_agent(Agent("c", AgentRole.EVALUATOR))
        coord.broadcast("coordinator", MessageType.STATUS, "hello all")

        # Each agent should have one message
        for name in ["a", "b", "c"]:
            agent = coord.get_agent(name)
            assert len(agent.inbox) == 1
            assert agent.inbox[0].content == "hello all"

        # All 3 messages should be in the log
        assert len(coord.message_log) == 3

    def test_get_message_log_unfiltered(self):
        coord = Coordinator()
        coord.add_agent(Agent("a", AgentRole.EXECUTOR))
        coord.add_agent(Agent("b", AgentRole.EXECUTOR))
        coord.broadcast("coord", MessageType.STATUS, "hi")
        log = coord.get_message_log()
        assert len(log) == 2

    def test_get_message_log_filtered(self):
        coord = Coordinator()
        alice = Agent("alice", AgentRole.PROFILER)
        bob = Agent("bob", AgentRole.EXECUTOR)
        coord.add_agent(alice)
        coord.add_agent(bob)

        # Send one message to alice, one to bob
        m1 = Message(
            id="m1", sender="coord", receiver="alice",
            msg_type=MessageType.TASK, content="t1",
        )
        m2 = Message(
            id="m2", sender="coord", receiver="bob",
            msg_type=MessageType.TASK, content="t2",
        )
        coord.send_message(m1)
        coord.send_message(m2)

        alice_log = coord.get_message_log("alice")
        assert len(alice_log) == 1
        assert alice_log[0].receiver == "alice"

        bob_log = coord.get_message_log("bob")
        assert len(bob_log) == 1
        assert bob_log[0].receiver == "bob"

    def test_create_default_agents(self):
        coord = Coordinator()
        coord.create_default_agents()
        assert len(coord.agents) == 7

        # Check expected names
        expected_names = {
            "profiler", "router",
            "executor_1", "executor_2", "executor_3",
            "evaluator", "sentinel",
        }
        assert set(coord.agents.keys()) == expected_names

        # Check roles
        assert coord.get_agent("profiler").role == AgentRole.PROFILER
        assert coord.get_agent("router").role == AgentRole.ROUTER
        assert coord.get_agent("executor_1").role == AgentRole.EXECUTOR
        assert coord.get_agent("evaluator").role == AgentRole.EVALUATOR
        assert coord.get_agent("sentinel").role == AgentRole.SENTINEL

    def test_create_default_agents_capabilities(self):
        coord = Coordinator()
        coord.create_default_agents()
        assert "icm" in coord.get_agent("evaluator").capabilities
        assert "early_warning" in coord.get_agent("sentinel").capabilities

    def test_summary(self):
        coord = Coordinator()
        coord.create_default_agents()
        s = coord.summary()
        assert s["n_agents"] == 7
        assert "agents" in s
        assert "total_messages" in s
        assert s["total_messages"] == 0

    def test_summary_after_messages(self):
        coord = Coordinator()
        coord.add_agent(Agent("a", AgentRole.EXECUTOR))
        coord.broadcast("coord", MessageType.STATUS, "hi")
        s = coord.summary()
        assert s["total_messages"] == 1
        assert s["messages_by_type"]["status"] == 1


# ===================================================================
# Work plan execution tests
# ===================================================================

class TestWorkPlanExecution:
    """Tests for Coordinator.execute_work_plan."""

    def test_simple_sequential(self):
        coord = Coordinator()
        a = Agent("a", AgentRole.PROFILER)
        b = Agent("b", AgentRole.ROUTER)

        # Handlers that produce results
        a.set_handler(lambda m: a.create_message(
            "coordinator", MessageType.RESULT, "profile_done"
        ))
        b.set_handler(lambda m: b.create_message(
            "coordinator", MessageType.RESULT, "route_done"
        ))

        coord.add_agent(a)
        coord.add_agent(b)

        plan = WorkPlan()
        plan.add_step("a", "profile")
        plan.add_step("b", "route", depends_on=["profile"])

        results = coord.execute_work_plan(plan)
        assert "profile" in results
        assert "route" in results
        assert results["profile"] == "profile_done"
        assert results["route"] == "route_done"

    def test_parallel_steps(self):
        coord = Coordinator()
        agents = {}
        for name in ["e1", "e2", "e3"]:
            ag = Agent(name, AgentRole.EXECUTOR)
            ag.set_handler(lambda m, n=name: ag.create_message(
                "coordinator", MessageType.RESULT, f"{n}_result"
            ))
            agents[name] = ag
            coord.add_agent(ag)

        plan = WorkPlan()
        plan.add_step("e1", "run_1")
        plan.add_step("e2", "run_2")
        plan.add_step("e3", "run_3")

        results = coord.execute_work_plan(plan)
        assert len(results) == 3
        for task in ["run_1", "run_2", "run_3"]:
            assert task in results

    def test_diamond_dag_execution(self):
        coord = Coordinator()
        for name in ["a", "b", "c", "d"]:
            ag = Agent(name, AgentRole.EXECUTOR)
            ag.set_handler(lambda m, n=name: ag.create_message(
                "coordinator", MessageType.RESULT, f"{n}_ok"
            ))
            coord.add_agent(ag)

        plan = WorkPlan()
        plan.add_step("a", "start")
        plan.add_step("b", "left", depends_on=["start"])
        plan.add_step("c", "right", depends_on=["start"])
        plan.add_step("d", "merge", depends_on=["left", "right"])

        results = coord.execute_work_plan(plan)
        assert len(results) == 4
        assert "merge" in results

    def test_execution_with_missing_agent(self):
        """Steps assigned to unknown agents should not block the plan."""
        coord = Coordinator()
        coord.add_agent(Agent("a", AgentRole.EXECUTOR))

        plan = WorkPlan()
        plan.add_step("a", "t1")
        plan.add_step("missing", "t2", depends_on=["t1"])

        results = coord.execute_work_plan(plan)
        assert "t1" in results
        assert "t2" in results
        assert results["t2"] is None

    def test_empty_plan_execution(self):
        coord = Coordinator()
        plan = WorkPlan()
        results = coord.execute_work_plan(plan)
        assert results == {}

    def test_agent_state_transitions(self):
        """Agent status should transition through working -> done."""
        coord = Coordinator()
        statuses_seen = []
        a = Agent("a", AgentRole.EXECUTOR)

        def handler(msg: Message) -> Message:
            statuses_seen.append(a.state.status)
            return a.create_message(
                "coordinator", MessageType.RESULT, "done"
            )

        a.set_handler(handler)
        coord.add_agent(a)

        plan = WorkPlan()
        plan.add_step("a", "task_1")
        coord.execute_work_plan(plan)

        # The handler was called while status was "working"
        assert "working" in statuses_seen
        # After execution the agent is "done"
        assert a.state.status == "done"


# ===================================================================
# Consensus execution tests
# ===================================================================

class TestConsensusExecution:
    """Tests for Coordinator.run_consensus."""

    def test_basic_consensus(self):
        coord = Coordinator()
        for name in ["v1", "v2", "v3"]:
            ag = Agent(name, AgentRole.EVALUATOR)
            ag.set_handler(lambda m, n=name: ag.create_message(
                "coordinator", MessageType.VOTE,
                "approve" if n != "v3" else "reject",
            ))
            coord.add_agent(ag)

        winner = coord.run_consensus("should we proceed?")
        assert winner == "approve"

    def test_consensus_subset_voters(self):
        coord = Coordinator()
        for name in ["v1", "v2", "v3"]:
            ag = Agent(name, AgentRole.EVALUATOR)
            ag.set_handler(lambda m, n=name: ag.create_message(
                "coordinator", MessageType.VOTE, n,
            ))
            coord.add_agent(ag)

        # Only v1 and v2 vote
        winner = coord.run_consensus("topic", voters=["v1", "v2"])
        assert winner in ("v1", "v2")

    def test_consensus_no_valid_voters_raises(self):
        coord = Coordinator()
        with pytest.raises(ValueError, match="No valid voters"):
            coord.run_consensus("topic", voters=["nonexistent"])


# ===================================================================
# Edge case and integration tests
# ===================================================================

class TestEdgeCases:
    """Edge cases and integration scenarios."""

    def test_agent_multiple_messages(self):
        agent = Agent("a", AgentRole.EXECUTOR)
        for i in range(5):
            msg = Message(
                id=f"m{i}", sender="coord", receiver="a",
                msg_type=MessageType.TASK, content=f"task_{i}",
            )
            agent.receive_message(msg)
        assert len(agent.inbox) == 5
        agent.process_inbox()
        assert len(agent.inbox) == 0
        assert len(agent.state.results) == 5

    def test_message_type_enum_values(self):
        assert MessageType.TASK.value == "task"
        assert MessageType.RESULT.value == "result"
        assert MessageType.VOTE.value == "vote"
        assert MessageType.ALERT.value == "alert"
        assert MessageType.STATUS.value == "status"

    def test_agent_role_enum_values(self):
        assert AgentRole.PROFILER.value == "profiler"
        assert AgentRole.ROUTER.value == "router"
        assert AgentRole.EXECUTOR.value == "executor"
        assert AgentRole.EVALUATOR.value == "evaluator"
        assert AgentRole.SENTINEL.value == "sentinel"

    def test_coordinator_overwrite_agent(self):
        """Adding an agent with an existing name should overwrite."""
        coord = Coordinator()
        old = Agent("a", AgentRole.EXECUTOR, ["old_cap"])
        new = Agent("a", AgentRole.PROFILER, ["new_cap"])
        coord.add_agent(old)
        coord.add_agent(new)
        assert coord.get_agent("a").role == AgentRole.PROFILER
        assert coord.get_agent("a").capabilities == ["new_cap"]

    def test_full_pipeline_simulation(self):
        """Simulate a full OS-MS pipeline: profile -> route -> execute -> evaluate."""
        coord = Coordinator()
        coord.create_default_agents()

        # Set handlers
        coord.get_agent("profiler").set_handler(
            lambda m: coord.get_agent("profiler").create_message(
                "coordinator", MessageType.RESULT,
                {"profile": "financial_energy"},
            )
        )
        coord.get_agent("router").set_handler(
            lambda m: coord.get_agent("router").create_message(
                "coordinator", MessageType.RESULT,
                {"kit": ["GBM", "SD", "ABM"]},
            )
        )
        for i in range(1, 4):
            name = f"executor_{i}"
            coord.get_agent(name).set_handler(
                lambda m, n=name: coord.get_agent(n).create_message(
                    "coordinator", MessageType.RESULT,
                    {"model": n, "predictions": [0.1, 0.2, 0.3]},
                )
            )
        coord.get_agent("evaluator").set_handler(
            lambda m: coord.get_agent("evaluator").create_message(
                "coordinator", MessageType.RESULT,
                {"icm": 0.85, "decision": "act"},
            )
        )
        coord.get_agent("sentinel").set_handler(
            lambda m: coord.get_agent("sentinel").create_message(
                "coordinator", MessageType.RESULT,
                {"alerts": []},
            )
        )

        # Build work plan
        plan = WorkPlan()
        plan.add_step("profiler", "profile_system")
        plan.add_step("router", "select_kit", depends_on=["profile_system"])
        plan.add_step("executor_1", "run_model_1", depends_on=["select_kit"])
        plan.add_step("executor_2", "run_model_2", depends_on=["select_kit"])
        plan.add_step("executor_3", "run_model_3", depends_on=["select_kit"])
        plan.add_step(
            "evaluator", "evaluate",
            depends_on=["run_model_1", "run_model_2", "run_model_3"],
        )
        plan.add_step("sentinel", "monitor", depends_on=["evaluate"])

        results = coord.execute_work_plan(plan)

        assert len(results) == 7
        assert results["profile_system"]["profile"] == "financial_energy"
        assert results["select_kit"]["kit"] == ["GBM", "SD", "ABM"]
        assert results["evaluate"]["icm"] == 0.85
        assert results["monitor"]["alerts"] == []

        # Verify audit trail
        summary = coord.summary()
        assert summary["total_messages"] > 0
        assert summary["n_agents"] == 7

    def test_message_log_includes_sender_filter(self):
        """Filtering by agent name should include messages sent BY that agent."""
        coord = Coordinator()
        alice = Agent("alice", AgentRole.PROFILER)
        bob = Agent("bob", AgentRole.EXECUTOR)
        coord.add_agent(alice)
        coord.add_agent(bob)

        msg = Message(
            id="m1", sender="alice", receiver="bob",
            msg_type=MessageType.TASK, content="hi",
        )
        coord.send_message(msg)

        # Alice is the sender, so she should appear in the filtered log
        alice_log = coord.get_message_log("alice")
        assert len(alice_log) == 1

        # Bob is the receiver, so he should also appear
        bob_log = coord.get_message_log("bob")
        assert len(bob_log) == 1

    def test_coordinator_repr(self):
        coord = Coordinator()
        coord.add_agent(Agent("a", AgentRole.EXECUTOR))
        r = repr(coord)
        assert "agents=1" in r
        assert "messages=0" in r
