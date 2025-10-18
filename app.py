import streamlit as st
import numpy as np
import os
import time
import pandas as pd
from Neural_Network import NeuralNetwork

st.set_page_config(page_title="Neural Network from Scratch")
st.title("Neural Network from Scratch")

# --- Training data ---
input_size = 2
hidden_size = 3
output_size = 1
epochs = 10000
learning_rate = 0.1

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

# --- Initialize model in session state ---
if "NN" not in st.session_state:
    st.session_state.NN = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
NN = st.session_state.NN

# --- Session states ---
if "training" not in st.session_state:
    st.session_state.training = False
if "trained" not in st.session_state:
    st.session_state.trained = False
if "testing" not in st.session_state:
    st.session_state.testing = False

# --- Image placeholder ---
image_path = os.path.join(os.getcwd(), "static", "NN.png")
image_placeholder = st.empty()
if not st.session_state.training and not st.session_state.trained:
    image_placeholder.image(image_path, caption="Neural Network Architecture")

# --- Live training placeholders ---
loss_placeholder = None
log_placeholder = None
epochs_list = []
losses_list = []
log_lines = []

# --- Button placeholder below everything ---
col1, col2, col3 = st.columns([1, 0.4, 1])
button_placeholder = col2.empty()

# --- Train button ---
if not st.session_state.training and not st.session_state.trained:
    if button_placeholder.button("Train", key="train_button"):
        st.session_state.training = True
        st.rerun()
elif st.session_state.training:
    button_placeholder.button("Training...", disabled=True, key="train_button")

# --- Live training loop ---
if st.session_state.training and not st.session_state.trained:
    # Hide image while training
    image_placeholder.empty()

    # Create chart/log layout
    col_chart, col_log = st.columns([2, 1])
    col_chart.subheader("Training Loss Over Epochs")
    loss_placeholder = col_chart.empty()
    log_placeholder = col_log.empty()

    for epoch, loss in NN.train(X, y, epochs):
        if epoch % 100 == 0:
            epochs_list.append(epoch)
            losses_list.append(loss)
            df = pd.DataFrame({"Epoch": epochs_list, "Loss": losses_list}).set_index("Epoch")
            loss_placeholder.line_chart(df, x_label="Epoch", y_label="Loss")
        if epoch % 1000 == 0:
            log_lines.append(f"Epoch {epoch}, Loss: {loss:.6f}")
            log_placeholder.text("\n".join(log_lines))
        time.sleep(0.001)

    # Hide train button after training
    button_placeholder.empty()
    st.session_state.trained = True
    st.session_state.training = False
    st.success(f"Training complete Final_Loss {loss:.6f}")

# --- Show Test button only after training ---
if st.session_state.trained and not st.session_state.testing:
    col1, col2, col3 = st.columns([1, 0.4, 1])
    test_placeholder = col2.empty()
    
    if test_placeholder.button("Test", key="test_button_unique"):
        st.session_state.testing = True
        test_placeholder.empty()  # remove the button
        image_placeholder.image(image_path, caption="Neural Network Architecture")  # show image


