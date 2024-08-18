def create_api_response(status=None, data=None, message=None, error=None):
    """
    This is a helper function to create a consistent JSON response that can be sent by the API.
    
    Args:
        status: The status of the API call. Should be one of the constants in this module.
        data: The data that was returned by the API call.
        error: The error that was returned by the API call.
    Returns: 
      A dictionary containing the status data and error if any
      :param status:
      :param data:
      :param message:
      :param error:
    """
    response = {"status": status}

    # Set the data of the response
    if data is not None:
        response["data"] = data

    if message is not None:
        response["message"] = message

    # Set the error message to the response.
    if error is not None:
        response["error"] = error

    return response
