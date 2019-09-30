from flask import Flask, request, redirect, render_template

app = Flask(__name__)

# Index route so when I browse to the url it doesn't 404
@app.route('/', methods=['Post', 'GET'])
def index():
    # Whenever someone wants to submit
    if request.method == 'POST':
        # TODO STUFF
        try:        
            # FOR POSTING DRAWN NUMBER
            # POST IT, COMPUTE, RETURN TO INDEX PAGE
            return redirect('/')
        # Error Handling
        except: 
            # If for some reason data couldn't be commit throw an error message
            return 'Issue with your number submission'
    else:
        # Base Page
        return render_template('index.html')

if __name__ == "__main__":
    # If theres any errors they'll pop up on the page
    app.run(debug=True)