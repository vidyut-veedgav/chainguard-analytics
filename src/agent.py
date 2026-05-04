"""Agent to handle conversational UI

Gemini agent has access to public API to answer questions from chat interface
"""

from google import genai
from google.genai import types

from src import api


# ---------------------------------------------------------------------------
# Function declarations — one per public method in src/api.py
# ---------------------------------------------------------------------------

get_account_risk_decl = types.FunctionDeclaration(
    name='get_account_risk',
    description='Get the churn probability and predicted label for a single account by id. Works for active and already-churned accounts.',
    parameters_json_schema={
        'type': 'object',
        'properties': {
            'account_id': {
                'type': 'string',
                'description': 'The account identifier to look up.',
            },
        },
        'required': ['account_id'],
    },
)

explain_account_decl = types.FunctionDeclaration(
    name='explain_account',
    description='Explain why an account has its churn score using Tree SHAP. Returns top drivers (push toward churn) and top protectors (push toward retention).',
    parameters_json_schema={
        'type': 'object',
        'properties': {
            'account_id': {
                'type': 'string',
                'description': 'The account identifier to explain.',
            },
            'top_k': {
                'type': 'integer',
                'description': 'Number of drivers and protectors to return (default 3).',
            },
        },
        'required': ['account_id'],
    },
)

top_risk_accounts_decl = types.FunctionDeclaration(
    name='top_risk_accounts',
    description='List the N currently-active accounts with the highest predicted churn probability. Already-churned accounts are excluded.',
    parameters_json_schema={
        'type': 'object',
        'properties': {
            'n': {
                'type': 'integer',
                'description': 'Number of accounts to return (default 5).',
            },
        },
    },
)

top_k_accounts_decl = types.FunctionDeclaration(
    name='top_k_accounts',
    description='Capacity-aware top-K view: returns the K riskiest active accounts plus the implied probability threshold at position K. Use when the user has a fixed CS capacity.',
    parameters_json_schema={
        'type': 'object',
        'properties': {
            'k': {
                'type': 'integer',
                'description': 'Number of accounts CS has capacity to handle.',
            },
        },
        'required': ['k'],
    },
)

feature_importance_decl = types.FunctionDeclaration(
    name='feature_importance',
    description='Return global feature importance (gain) from the trained XGBoost model, descending.',
    parameters_json_schema={
        'type': 'object',
        'properties': {
            'top_k': {
                'type': 'integer',
                'description': 'Number of top features to return (default 10).',
            },
        },
    },
)

portfolio_summary_decl = types.FunctionDeclaration(
    name='portfolio_summary',
    description='Dashboard rollups across the active book (counts, high-risk count, average probability) plus held-out model metrics (precision, recall, F1, F2, AP).',
    parameters_json_schema={
        'type': 'object',
        'properties': {},
    },
)

probability_distribution_decl = types.FunctionDeclaration(
    name='probability_distribution',
    description='Histogram of predicted churn probabilities across active accounts. Returns bin edges and counts for plotting.',
    parameters_json_schema={
        'type': 'object',
        'properties': {
            'bins': {
                'type': 'integer',
                'description': 'Number of histogram bins (default 20).',
            },
        },
    },
)


tool = types.Tool(function_declarations=[
    get_account_risk_decl,
    explain_account_decl,
    top_risk_accounts_decl,
    top_k_accounts_decl,
    feature_importance_decl,
    portfolio_summary_decl,
    probability_distribution_decl,
])


# ---------------------------------------------------------------------------
# Dispatcher — maps function name to the real implementation 
# ---------------------------------------------------------------------------

DISPATCH = {
    'get_account_risk':         api.get_account_risk,
    'explain_account':          api.explain_account,
    'top_risk_accounts':        api.top_risk_accounts,
    'top_k_accounts':           api.top_k_accounts,
    'feature_importance':       api.feature_importance,
    'portfolio_summary':        api.portfolio_summary,
    'probability_distribution': api.probability_distribution,
}


# ---------------------------------------------------------------------------
# System instructions
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """

You are a helpful assistant that helps users identify, rank, and explain churn risk among their customer accounts.
You have access to various tools (functions built from an underlying machine learning system) that you must use to do this. 

Make sure to use the tool that best matches the user's need. 

For example, if a user asks: "What is the probability that Account ACC000456 churns?", you should use the 'get_account_risk' tool. 

Do NOT hallucinate. All of your answers to the users MUST be grounded in the return values from tools you decide to you.

You must justify all of your responses to the user. You provide clear and interpretable explanations to both technical and nontechnical stakeholders. 

You also talk like a pirate.
"""

# Lazy client — Vertex auth resolution can block at construction, so we defer
# it to the first ask() call. Dashboard consumers that never call ask() pay
# nothing.

_client = None


def _get_client():
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project='kepler-labs-491817',
            location='us-central1',
        )
    return _client


MODEL = 'gemini-2.5-flash'

def ask(message: str, chat=None) -> tuple[str, object]:
    # Send a user message and resolve any tool calls until an answer comes back
    if chat is None:
        chat = _get_client().chats.create(
            model=MODEL,
            config=types.GenerateContentConfig(
                tools=[tool],
                system_instruction=SYSTEM_INSTRUCTIONS
            ),
        )
    # Response and tool use logic
    response = chat.send_message(message)
    while True:
        calls = [p.function_call for p in response.candidates[0].content.parts if p.function_call]
        if not calls:
            return response.text, chat

        responses = []
        for call in calls:
            result = DISPATCH[call.name](**(call.args or {}))
            responses.append(types.Part.from_function_response(name=call.name, response={'result': result}))

        response = chat.send_message(responses)
