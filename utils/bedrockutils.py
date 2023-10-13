import boto3

def invoke_model(modelId):
    #modelId = 'anthropic.claude-v2' # change this to use a different version from the model provider
    accept = 'application/json'
    contentType = 'application/json'

    response = boto3_bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())

    print_ww(response_body.get('completion'))
    
