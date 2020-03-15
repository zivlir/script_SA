def predict(text, with_neu=True): # requires string input.
    start_at = time.time()
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=seq_len)
    score = model.predict([x_test])[0]
    label = decode_sentiment(score, with_neu=with_neu)
    score = float(score)
    return {print(f"label: {label}, score: {score}, calculation duration: {time.time()-start_at}.")}

print('Welcome!')
print('Script for post toxicity estimation is evaluated. Enter your text below for sentiment estimation and press ENTER.')
post_text = str(input('Your text: '))
print(predict(post_text))
print(f'In terminal type predict(text) and pass string to it for the next prediction. Type file_sanalysis(filepath) to make several predictions.  Local time: {datetime.datetime.now().time()}')

def file_sanalysis(filepath):
    file = open(filepath, 'rt')
    c = 0
    n = 0
    processed_file = file.readlines()
    for text in processed_file:
            n += 1
            print(n,'', text)
            print(predict(text))
            c += 1
            print(f'\n')
    print(f'Total comments analyzed: {c}')

file_sanalysis('../comment.txt')


