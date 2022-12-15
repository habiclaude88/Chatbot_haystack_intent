from flask import Flask, render_template, request,jsonify
from chat import get_response


app = Flask(__name__)

# create two routes
@app.get("/")
def index_get():
    return render_template("base.html")

@app.post("/predict")
def predict():
    text = request.get_json().get("message")
    #check if text is valid (I let it for you)
    response = get_response(text)
    # we jsonify our response
    message = {"answer":response}
    return jsonify(message)


# if __name__=='__main__': we can start our app
if __name__=='__main__':
    app.run(debug=True)#debug is equal to True for testing