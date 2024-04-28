# flake8: noqa

from langchain_core.prompts.prompt import PromptTemplate

API_PLANNER_PROMPT = """You are a planner that plans a sequence of API calls to assist with user queries against an API.

You should:
1) evaluate whether the user query can be solved by the API documentated below. If no, say why.
2) if yes, generate a plan of API calls and say what they are doing step by step.
3) If the plan includes a DELETE call, you should always return a request for user authorization first unless the user has specifically asked to delete something.

You should only use API endpoints documented below ("Endpoints you can use:").
You can only use the DELETE tool if the user has specifically asked to delete something. Otherwise, you should return a request for authorization from the user first.
Some user queries can be resolved in a single API call, but some will require several API calls.
The plan will be passed to an API controller that can format it into web requests and return the responses.

----

Here are some examples:

Example endpoints:
GET /user to get information about the current user
GET /products/search search across products
POST /users/{id}/cart to add products to a user's cart
PATCH /users/{id}/cart to update a user's cart
PUT /users/{id}/coupon to apply coupon to a user's cart
DELETE /users/{id}/cart to delete a user's cart

User query: tell me a joke
Plan: Sorry, this API's domain is shopping, not comedy.

User query: I want to buy a couch
Plan: 1. GET /products with a query param to search for couches
2. GET /user to find the user's id
3. POST /users/{id}/cart to add a couch to the user's cart

User query: I want to add a lamp to my cart
Plan: 1. GET /products with a query param to search for lamps
2. GET /user to find the user's id
3. PATCH /users/{id}/cart to add a lamp to the user's cart

User query: I want to add a coupon to my cart
Plan: 1. GET /user to find the user's id
2. PUT /users/{id}/coupon to apply the coupon

User query: I want to delete my cart
Plan: 1. GET /user to find the user's id
2. DELETE required. Did user specify DELETE or previously authorize? No, ask for authorization.
3. DELETE /users/{id}/cart to delete the user's cart

User query: I want to start a new cart
Plan: 1. GET /user to find the user's id
2. DELETE required. Did user specify DELETE or previously authorize? No, ask for authorization.
3. DELETE /users/{id}/cart to delete the user's cart

----

Here are endpoints you can use. Do not reference any of the endpoints above.

{endpoints}

----

User query: {query}
Plan:"""
API_PLANNER_TOOL_NAME = "api_planner"
API_PLANNER_TOOL_DESCRIPTION = f"Can be used to generate the right API calls to assist with a user query, like {API_PLANNER_TOOL_NAME}(query). Should always be called before trying to call the API controller."

REQUESTS_GET_TOOL_DESCRIPTION = """Use this to GET content from a website.
Input to the tool should be a json string with 3 keys: "url", "params" and "output_instructions".
The value of "url" should be a string. 
The value of "params" should be a dict of the needed and available parameters from the OpenAPI spec related to the endpoint. 
If parameters are not needed, or not available, leave it empty.
The value of "output_instructions" should be instructions on what information to extract from the response, 
for example the id(s) for a resource(s) that the GET request fetches.
"""

PARSING_GET_PROMPT = PromptTemplate(
    template="""Here is an API response:\n\n{response}\n\n====
Your task is to extract some information according to these instructions: {instructions}
When working with API objects, you should usually use ids over names.
If the response indicates an error, you should instead output a summary of the error.

Output:""",
    input_variables=["response", "instructions"],
)

REQUESTS_PROMPT_TEMPLATE = PromptTemplate(
    template="""Use this tool to make an API request.
Input to the tool should be a json string with 3 keys: 'url', 'method', and 'data'.
The value of 'url' should be a string.
The value of 'method' should be a string, either 'GET', 'POST', 'PATCH', 'PUT', or 'DELETE'.
The value of 'data' should be a dictionary of key-value pairs to send with the request.
If the request does not require data, leave it empty.
Always use double quotes for strings in the json string.

Here is the input: {input}

{}
""",
    input_variables=["input"],
)

class APIRequestTool:
    def __init__(self, api_base_url, api_docs):
        self.api_base_url = api_base_url
        self.api_docs = api_docs

    def request(self, input):
        prompt = REQUESTS_PROMPT_TEMPLATE.format(input=input)
        # Call a language model to generate the API request
        # based on the prompt and the API documentation
        # Return the generated API request
        return generated_api_request

class APIController:
    def __init__(self, api_request_tool):
        self.api_request_tool = api_request_tool

    def execute_plan(self, plan):
        # Execute each step in the plan using the API request tool
        # and return the final response
        for step in plan:
            self.api_request_tool.request(step)
        return final_response

class APIOrchestrator:
    def __init__(self, api_planner, api_controller):
        self.api_planner = api_planner
        self.api_controller = api_controller

    def assist_with_query(self, query):
        plan = self.api_planner.plan(query)
        response = self.api_controller.execute_plan(plan)
        return response
