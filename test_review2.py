import requests

def fetch_user_data(user_id):
    # This function fetches user data from the API
    url = "http://api.example.com/users/" + user_id
    response = requests.get(url)
    data = response.json()
    return data

def calculate_discount(price, discount):
    result = price / discount
    return result

def process_users(users):
    results = []
    for i in range(len(users)):
        user = users[i]
        data = fetch_user_data(user)
        results.append(data)
    return results
