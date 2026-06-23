# Market Maker RL

A reinforcement learning project that trains an AI agent to act as a market maker in a simulated financial market.

The agent learns how to place buy and sell quotes based on market conditions such as price movements, volatility, order flow, and inventory levels. Its objective is to maximize profit from the bid-ask spread while managing inventory risk.

The project includes:

* A custom Actor-Critic reinforcement learning model built with NumPy
* A simulated market-making environment
* Synthetic and CSV-based data support
* Real-time training visualizations using Streamlit and Plotly
* Performance evaluation using metrics such as PnL and win rate

### Tech Stack

* Python
* NumPy
* Pandas
* Streamlit
* Plotly

### Running the Project

Install dependencies:

```bash
pip install streamlit numpy pandas plotly
```

Run the application:

```bash
streamlit run app.py
```

### Future Improvements

* Full PPO implementation
* More realistic market simulations
* Support for live market data
* Additional risk management features

This project was developed to explore the application of reinforcement learning techniques in quantitative finance and algorithmic trading.
