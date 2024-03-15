import mysql.connector
from flask import session, render_template, request, Flask, redirect, url_for
import hashlib
import nltk
from nltk.tokenize import word_tokenize
from googletrans import Translator
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import re
import emoji
import cleantext

translator=Translator()

def prepare_data(data):
    translated=[]
    for i in data:
        trans = translator.translate(i)
        translated.append(cleaned_text(trans.text))
        translated=[re.sub(r"['\[\],]", "", str(t)).strip() for t in translated]
    return translated

def cleaned_text(text):
    text= emoji.replace_emoji(text, replace='')
    return cleantext.clean_words(text,stemming=False,)
def generate_unique_key(sentence):
    # Convert the sentence to bytes (required by hashlib)
    sentence_bytes = sentence.encode('utf-8')
    
    # Use a hashing algorithm (e.g., SHA-256) to generate the unique key
    hashed_key = hashlib.sha256(sentence_bytes).hexdigest()
    
    return hashed_key

nltk.download(['punkt','stopwords','twitter_samples'])
removeables = {':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3', ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';(', '(', ')','*','=','!',"'",'&amp;',',',':','.','-','_','0','1','2','3','4','5','6','7','8','9',
    'via','RT','\n','#','@','http'}

tweets = [[1,t] for t in nltk.corpus.twitter_samples.strings('positive_tweets.json')]
tweets1 = [[-1,t] for t in nltk.corpus.twitter_samples.strings('negative_tweets.json')]
tweets=tweets+tweets1

def give_emoji_free_text(text):
    return emoji.replace_emoji(text, replace='')

for i in range(0,len(tweets)):
    tweets[i][1]=give_emoji_free_text(tweets[i][1])


for i in range(0,len(tweets)):
    for t in removeables:
        if t=='#':
             pattern = r'\#\w+'
             tweets[i][1] = re.sub(pattern, '', tweets[i][1]).strip().lower()
        elif t=='@':
            pattern = r'\@\w+'
            tweets[i][1] = re.sub(pattern, '', tweets[i][1]).strip().lower()
        elif t=='http':
            pattern = r'http\S+'
            tweets[i][1] = re.sub(pattern, '', tweets[i][1]).strip().lower()
        elif t in tweets[i][1]:
            tweets[i][1]=tweets[i][1].replace(t,"").strip().lower()




def tokenize_text(text):
    return word_tokenize(text)

# Stopwords removal
stop_words = set(stopwords.words('english'))
def remove_stopwords(tokens):
    return [word for word in tokens if word not in stop_words]

labels=[]
# Apply feature extraction functions to each text in the dataset
processed_text_data = []
for text in tweets:
    labels.append(text[0])
    tokens = tokenize_text(text[1])
    tokens = remove_stopwords(tokens)
    processed_text_data.append(tokens)


for i in range(len(processed_text_data)):
    temp=""
    for j in processed_text_data[i]:
        temp=temp+ ' '+j
    tweets[i][1]=temp


X = [" ".join(tokens) for tokens in processed_text_data]  # Convert tokenized text data into strings
y = labels  # Replace labels_list with your actual labels

# Initialize CountVectorizer to convert tokenized text into feature vectors
vectorizer = CountVectorizer()

# Fit and transform the tokenized text into feature vectors
X = vectorizer.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the sentiment analysis model 
model = RandomForestClassifier()
model.fit(X_train, y_train)

app = Flask(__name__)
mydb = mysql.connector.connect(
    host="sql.freedb.tech",
    user="freedb_root14255",
    passwd="PD?NTpK?EP$8w6@",
    database="freedb_mydb14255"
)
app.secret_key="hitherethisisnotsecretkey"
my_cursor = mydb.cursor(buffered=True)

@app.route('/', methods=['POST', 'GET'])
def home():     
    return render_template('index.html')


@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == "POST":
        if 'email' in request.form:
            uname = request.form['name']
            email = request.form['email']
            password = request.form['password']
            query = f"insert into users(name, email, password) values('{uname}', '{email}','{password}');"
            my_cursor.execute(query)
            mydb.commit()
            return render_template('login.html')
        else:
            query = f"select id, name, password from users where name='{request.form['name']}' and password='{request.form['password']}'"
            session['name']=request.form['name']
            my_cursor.execute(query)
            result= [i for i in my_cursor]
            if result:
                return redirect(url_for('form'))
    return render_template('login.html')


@app.route('/form', methods=['POST', 'GET'])
def form():
    link=""
    if request.method == 'POST':
        event = request.form['event_name']
        question = request.form['question']
        session['question']=question
        session['event']=event
        return redirect(url_for('response'))
    return render_template("form.html")


@app.route('/response', methods=['POST', 'GET'])
def response():
    if request.method == 'POST':
        response = request.form['response']
        query = f"insert into event_responses(name, event_name, responses) values('{session['name']}', '{session['event']}','{response}');"
        my_cursor.execute(query)
        mydb.commit()
        return redirect(url_for('thanks'))
    return render_template('response.html')


@app.route('/thanks', methods=['POST', 'GET'])
def thanks():
    query= f"select responses from event_responses where event_name='{session['event']}'"
    my_cursor.execute(query)
    data=[i for i in my_cursor]
    prepared_data = prepare_data(data)
    new_input_features = vectorizer.transform(prepared_data)
    result = model.predict(new_input_features)
    sentiment=sum(result)/len(result)
    return render_template("thanks.html",sentiment=round(sentiment,3))


if __name__ == '__main__':
    app.run(debug=True, port=5001)