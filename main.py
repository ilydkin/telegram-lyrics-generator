'''Telegram bot generating lyrics with a pre-trained model'''
from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os

from lyrics import util


model = os.path.join(os.path.expanduser('~'), 'telegram-lyrics-generator', 'export', 'model.h5')
tokenizer = os.path.join(os.path.expanduser('~'), 'telegram-lyrics-generator', 'export', 'tokenizer.pickle')
tg_token ='YourToken'


def load_model (model_filename):
    return tf.keras.models.load_model(
        model_filename, custom_objects={"KerasLayer": hub.KerasLayer}
    )


def softmax_sampling (probabilities, randomness, seed=None):
    """Returns the index of the highest value from a softmax vector,
    with a bit of randomness based on the probabilities returned.
    """
    if seed:
        np.random.seed(seed)
    if randomness == 0:
        return np.argmax(probabilities)
    probabilities = np.asarray(probabilities).astype("float64")
    probabilities = np.log(probabilities) / randomness
    exp_probabilities = np.exp(probabilities)
    probabilities = exp_probabilities / np.sum(exp_probabilities)
    return np.argmax(np.random.multinomial(1, probabilities, 1))


def generate_lyrics (model, tokenizer, text_seed, song_length, randomness=0, seed=None):
    input_shape = model.inputs[0].shape
    seq_length = -1
    if len(input_shape) >= 2:
        print("Using integer sequences")
        seq_length = int(input_shape[1])
    else:
        print("Using string sequences")

    # Create a reverse lookup index for integers to words
    rev = {v: k for k, v in tokenizer.word_index.items()}

    spacer = "" if tokenizer.char_level else " "

    text_output = tokenizer.texts_to_sequences([text_seed])[0]
    text_output_str = spacer.join(rev.get(word) for word in text_output)
    while len(text_output) < song_length:
        if seq_length != -1:
            padded = tf.keras.preprocessing.sequence.pad_sequences(
                [text_output], maxlen=seq_length, padding="post"
            )
        else:
            padded = np.array([text_output_str])
        next_word = model.predict_on_batch(padded)
        next_word = softmax_sampling(next_word[0], randomness, seed=seed)
        text_output.append(next_word)
        text_output_str += f"{spacer}{rev.get(next_word)}"
    return text_output, text_output_str


def lyrics (text, length, model, tokenizer):
    model = load_model(model)
    tokenizer = util.load_tokenizer(tokenizer)
    seed = (np.random.randint(np.iinfo(np.int32).max))
    raw, output_text = generate_lyrics (model, tokenizer, text, length, 0.0, seed=seed)
    return output_text


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Be my guest, the lonely stranger {user.mention_html()}! "
        rf"Manowar Oracle has been here awaiting your appearance. "
        rf"Ask what you dare and discern the fate in the gloom of future. "
        rf"Ask with caution, for much grief be there in much knowledge.",
        reply_markup=ForceReply(selective=True),
    )


async def predict_lyrics_tg (update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    length = 50
    answer = lyrics(update.message.text, length, model, tokenizer)
    print ('prediction:', answer)
    await update.message.reply_text(answer)


def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(tg_token).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))

    # on non command i.e message - echo the message on Telegram
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, predict_lyrics_tg))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()