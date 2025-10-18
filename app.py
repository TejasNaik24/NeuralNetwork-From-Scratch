import streamlit as st
import numpy as np
import os
import time
import pandas as pd
from Neural_Network import NeuralNetwork

st.set_page_config(
    page_title="Neural Network from Scratch",
    page_icon="static/NN.png",
)
st.title("Neural Network from Scratch")

if "training" not in st.session_state:
    st.session_state.training = False
if "trained" not in st.session_state:
    st.session_state.trained = False
if "testing" not in st.session_state:
    st.session_state.testing = False
if "epochs_list" not in st.session_state:
    st.session_state.epochs_list = []
if "losses_list" not in st.session_state:
    st.session_state.losses_list = []
if "log_lines" not in st.session_state:
    st.session_state.log_lines = []

input_size, hidden_size, output_size = 2, 3, 1
epochs, learning_rate = 10000, 0.1
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
NN = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)

image_path = os.path.join(os.getcwd(), "static", "NN.png")
image_placeholder = st.empty()

if not st.session_state.training and not st.session_state.trained:
    image_placeholder.image(image_path, caption="Neural Network Architecture")
    col1, col2, col3 = st.columns([1, 0.4, 1])
    if col2.button("Train", key="train_button"):
        st.session_state.training = True
        st.rerun()

if st.session_state.training or st.session_state.trained:
    image_placeholder.empty()
    
    col_chart, col_log = st.columns([2, 1])
    
    if st.session_state.training:
        col_chart.subheader("Training Loss Over Epochs")
        loss_placeholder = col_chart.empty()
        log_placeholder = col_log.empty()

        for epoch, loss in NN.train(X, y, epochs):
            if epoch % 100 == 0:
                st.session_state.epochs_list.append(epoch)
                st.session_state.losses_list.append(loss)
                df = pd.DataFrame({"Epoch": st.session_state.epochs_list,
                                   "Loss": st.session_state.losses_list}).set_index("Epoch")
                loss_placeholder.line_chart(df, x_label="Epoch", y_label="Loss")
            if epoch % 1000 == 0 or epoch == epochs-1:
                st.session_state.log_lines.append(f"Epoch {epoch}, Loss: {loss:.6f}")
                log_placeholder.text("\n".join(st.session_state.log_lines))
            time.sleep(0.001)

        st.session_state.training = False
        st.session_state.trained = True

        st.success(f"Training complete! Final Loss: {loss:.6f}")

    if st.session_state.trained and not st.session_state.testing:
        col1, col2, col3 = st.columns([1, 0.4, 1])
        test_placeholder = col2.empty()
        if test_placeholder.button("Test", key="test_button"):
            st.session_state.testing = True
            test_placeholder.empty()

if st.session_state.testing:
    tab1, tab2 = st.tabs(["Testing Mode", "Training Info"])
    with tab1:
        st.subheader("Testing Mode")
        st.image(image_path, caption="Neural Network Architecture")

        col1, col2 = st.columns(2)
        input1 = col1.number_input("Input 1", min_value=0, max_value=1, step=1)
        input2 = col2.number_input("Input 2", min_value=0, max_value=1, step=1)

        col_left, col_center, col_right = st.columns([1, 0.4, 1])
        if col_center.button("Run NN", key="run_nn_button"):
            user_input = np.array([[input1, input2]])
            prediction, confidence = NN.predict_with_confidence(user_input)
            st.success(f"Prediction for [{input1}, {input2}] is: {prediction} ({confidence*100:.2f}% confident)")


    with tab2:
        col_chart, col_log = st.columns([2, 1])
        if st.session_state.epochs_list and st.session_state.losses_list:
            df = pd.DataFrame({"Epoch": st.session_state.epochs_list,
                               "Loss": st.session_state.losses_list}).set_index("Epoch")
            col_chart.line_chart(df, x_label="Epoch", y_label="Loss")
        if st.session_state.log_lines:
            col_log.text("\n".join(st.session_state.log_lines))
