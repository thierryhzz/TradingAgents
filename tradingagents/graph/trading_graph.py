# TradingAgents/graph/trading_graph.py

import os
from pathlib import Path
import json
from datetime import date
from typing import Dict, Any, Tuple, List, Optional
import time

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# Ensure .env is loaded before any LLM initialization
load_dotenv()

# Configuration for API quota handling
API_QUOTA_RETRY_CONFIG = {
    "max_retries": 3,
    "initial_wait_seconds": 60,  # Wait 1 minute before first retry
    "backoff_multiplier": 2,     # Exponential backoff: 1min, 2min, 4min
}

from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_sentiment,
    get_insider_transactions,
    get_global_news
)

from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


def retry_on_quota_error(func):
    """
    Decorator to retry API calls with exponential backoff when quota is exceeded.
    Catches RateLimitError (429) and waits before retrying.
    """
    def wrapper(*args, **kwargs):
        wait_time = API_QUOTA_RETRY_CONFIG["initial_wait_seconds"]
        max_retries = API_QUOTA_RETRY_CONFIG["max_retries"]
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Check if it's a quota/rate limit error
                if "429" in str(e) or "insufficient_quota" in str(e) or "exceeded your current quota" in str(e):
                    if attempt < max_retries:
                        print(f"\n⚠️  API Quota Limit Hit (429). Waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        wait_time *= API_QUOTA_RETRY_CONFIG["backoff_multiplier"]
                    else:
                        print(f"\n❌ API Quota Limit exceeded after {max_retries} retries. Please check your OpenAI billing at https://platform.openai.com/account/billing/overview")
                        raise
                else:
                    raise
    
    return wrapper


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG

        # Update the interface's config
        set_config(self.config)

        # Create necessary directories
        os.makedirs(
            os.path.join(self.config["project_dir"], "dataflows/data_cache"),
            exist_ok=True,
        )

        # Initialize LLMs
        if self.config["llm_provider"].lower() == "openai" or self.config["llm_provider"] == "ollama" or self.config["llm_provider"] == "openrouter":
            # Prefer explicit API key from environment to avoid relying on implicit global state
            openai_api_key = os.getenv("OPENAI_API_KEY")
            self.deep_thinking_llm = ChatOpenAI(
                model=self.config["deep_think_llm"],
                base_url=self.config["backend_url"],
                api_key=openai_api_key,
            )
            self.quick_thinking_llm = ChatOpenAI(
                model=self.config["quick_think_llm"],
                base_url=self.config["backend_url"],
                api_key=openai_api_key,
            )
        elif self.config["llm_provider"].lower() == "anthropic":
            self.deep_thinking_llm = ChatAnthropic(model=self.config["deep_think_llm"], base_url=self.config["backend_url"])
            self.quick_thinking_llm = ChatAnthropic(model=self.config["quick_think_llm"], base_url=self.config["backend_url"])
        elif self.config["llm_provider"].lower() == "google":
            self.deep_thinking_llm = ChatGoogleGenerativeAI(model=self.config["deep_think_llm"])
            self.quick_thinking_llm = ChatGoogleGenerativeAI(model=self.config["quick_think_llm"])
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config['llm_provider']}")
        
        # Initialize memories
        self.bull_memory = FinancialSituationMemory("bull_memory", self.config)
        self.bear_memory = FinancialSituationMemory("bear_memory", self.config)
        self.trader_memory = FinancialSituationMemory("trader_memory", self.config)
        self.invest_judge_memory = FinancialSituationMemory("invest_judge_memory", self.config)
        self.risk_manager_memory = FinancialSituationMemory("risk_manager_memory", self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic()
        self.graph_setup = GraphSetup(
            self.quick_thinking_llm,
            self.deep_thinking_llm,
            self.tool_nodes,
            self.bull_memory,
            self.bear_memory,
            self.trader_memory,
            self.invest_judge_memory,
            self.risk_manager_memory,
            self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph
        self.graph = self.graph_setup.setup_graph(selected_analysts)

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_sentiment,
                    get_insider_transactions,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                ]
            ),
        }

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date."""

        self.ticker = company_name

        # Initialize state
        init_agent_state = self.propagator.create_initial_state(
            company_name, trade_date
        )
        args = self.propagator.get_graph_args()

        if self.debug:
            # Debug mode with tracing - also with retry logic
            final_state = self._stream_with_quota_retry(init_agent_state, args)
        else:
            # Standard mode without tracing - with quota error handling
            final_state = self._invoke_with_quota_retry(init_agent_state, args)

        # Store current state for reflection
        self.curr_state = final_state

        # Log state
        self._log_state(trade_date, final_state)

        # Return decision and processed signal
        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _stream_with_quota_retry(self, init_agent_state, args):
        """
        Stream the graph (debug mode) with automatic retry on OpenAI quota errors.
        """
        if not self.config.get("enable_quota_retry", True):
            # Retry disabled, stream normally
            trace = []
            response_content = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)
                    if hasattr(chunk["messages"][-1], "content"):
                        response_content.append(str(chunk["messages"][-1].content))
            
            full_response = " ".join(response_content)
            first_100_words = " ".join(full_response.split()[:100])
            print(f"\n✅ Successfully received valid response from OpenAI")
            print(f"First 100 words: {first_100_words}...\n")
            return trace[-1]
        
        max_retries = self.config.get("max_quota_retries", 3)
        wait_time = self.config.get("quota_retry_wait_seconds", 60)
        
        for attempt in range(max_retries + 1):
            response_content = []
            try:
                trace = []
                for chunk in self.graph.stream(init_agent_state, **args):
                    if len(chunk["messages"]) == 0:
                        pass
                    else:
                        chunk["messages"][-1].pretty_print()
                        trace.append(chunk)
                        if hasattr(chunk["messages"][-1], "content"):
                            response_content.append(str(chunk["messages"][-1].content))
                
                # Success! Print confirmation message with response preview
                full_response = " ".join(response_content)
                first_100_words = " ".join(full_response.split()[:100])
                print(f"\n✅ Successfully received valid response from OpenAI")
                print(f"First 100 words: {first_100_words}...\n")
                return trace[-1]
            except Exception as e:
                error_str = str(e).lower()
                error_type = type(e).__name__.lower()
                
                is_quota_error = (
                    "429" in error_str or
                    "insufficient_quota" in error_str or
                    "exceeded your current quota" in error_str or
                    "quota" in error_str or
                    "rate limit" in error_str or
                    "ratelimiterror" in error_type
                )
                
                if is_quota_error:
                    if attempt < max_retries:
                        print(f"\n⚠️  OpenAI API Quota Limit Hit - Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                        print(f"   (Error: {str(e)[:100]}...)")
                        time.sleep(wait_time)
                        wait_time *= 2
                    else:
                        print(f"\n❌ OpenAI API Quota still exceeded after {max_retries} retries.")
                        print(f"❌ Please check your billing: https://platform.openai.com/account/billing/overview")
                        print(f"❌ Original error: {str(e)}")
                        raise
                else:
                    raise

    def _invoke_with_quota_retry(self, init_agent_state, args):
        """
        Invoke the graph with automatic retry on OpenAI quota errors (429).
        Implements exponential backoff to wait for quota to reset.
        """
        if not self.config.get("enable_quota_retry", True):
            # Retry disabled, invoke normally
            return self.graph.invoke(init_agent_state, **args)
        
        max_retries = self.config.get("max_quota_retries", 3)
        wait_time = self.config.get("quota_retry_wait_seconds", 60)
        
        for attempt in range(max_retries + 1):
            try:
                result = self.graph.invoke(init_agent_state, **args)
                # Success! Print confirmation message with response preview
                response_text = ""
                if "final_trade_decision" in result and result["final_trade_decision"]:
                    response_text = str(result["final_trade_decision"])
                elif len(result.get("messages", [])) > 0:
                    response_text = str(result["messages"][-1].content if hasattr(result["messages"][-1], "content") else result["messages"][-1])
                
                first_100_words = " ".join(response_text.split()[:100])
                print(f"\n✅ Successfully received valid response from OpenAI")
                print(f"First 100 words: {first_100_words}...\n")
                return result
            except Exception as e:
                error_str = str(e).lower()
                error_type = type(e).__name__.lower()
                
                # Check if it's a quota error with more flexible matching
                is_quota_error = (
                    "429" in error_str or
                    "insufficient_quota" in error_str or
                    "exceeded your current quota" in error_str or
                    "quota" in error_str or
                    "rate limit" in error_str or
                    "ratelimiterror" in error_type
                )
                
                if is_quota_error:
                    if attempt < max_retries:
                        print(f"\n⚠️  OpenAI API Quota Limit Hit - Waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                        print(f"   (Error: {str(e)[:100]}...)")
                        time.sleep(wait_time)
                        wait_time *= 2  # Exponential backoff: double the wait time
                    else:
                        print(f"\n❌ OpenAI API Quota still exceeded after {max_retries} retries.")
                        print(f"❌ Please check your billing: https://platform.openai.com/account/billing/overview")
                        print(f"❌ Original error: {str(e)}")
                        raise
                else:
                    # Not a quota error, raise immediately
                    raise

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "risky_history": final_state["risk_debate_state"]["risky_history"],
                "safe_history": final_state["risk_debate_state"]["safe_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file
        directory = Path(f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/")
        directory.mkdir(parents=True, exist_ok=True)

        with open(
            f"eval_results/{self.ticker}/TradingAgentsStrategy_logs/full_states_log_{trade_date}.json",
            "w",
        ) as f:
            json.dump(self.log_states_dict, f, indent=4)

    def reflect_and_remember(self, returns_losses):
        """Reflect on decisions and update memory based on returns."""
        self.reflector.reflect_bull_researcher(
            self.curr_state, returns_losses, self.bull_memory
        )
        self.reflector.reflect_bear_researcher(
            self.curr_state, returns_losses, self.bear_memory
        )
        self.reflector.reflect_trader(
            self.curr_state, returns_losses, self.trader_memory
        )
        self.reflector.reflect_invest_judge(
            self.curr_state, returns_losses, self.invest_judge_memory
        )
        self.reflector.reflect_risk_manager(
            self.curr_state, returns_losses, self.risk_manager_memory
        )

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
