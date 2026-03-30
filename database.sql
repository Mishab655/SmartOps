CREATE TABLE category_forecast (
    id SERIAL PRIMARY KEY,
    forecast_date DATE NOT NULL,
    product_category VARCHAR(100) NOT NULL,
    actual_sales NUMERIC,
    predicted_sales NUMERIC NOT NULL,
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE customer_churn_prediction (
    id SERIAL PRIMARY KEY,
    customer_unique_id VARCHAR(50) NOT NULL,
    recency INT,
    frequency INT,
    monetary NUMERIC,
    r_score INT,
    f_score INT,
    m_score INT,
    rfm_risk_score INT NOT NULL,
    churn_risk VARCHAR(10) NOT NULL, -- 'High', 'Medium', 'Low'
    model_version VARCHAR(50),
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE category_sentiment_summary (
    id SERIAL PRIMARY KEY,
    product_category VARCHAR(100) NOT NULL,
    avg_review_score NUMERIC,
    review_count INT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE image_prediction_log (
    id SERIAL PRIMARY KEY,
    image_id VARCHAR(100),
    predicted_amazon_category VARCHAR(100),
    mapped_olist_category VARCHAR(100),
    confidence_score NUMERIC,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE decision_action_log (
    id SERIAL PRIMARY KEY,
    action_type VARCHAR(100) NOT NULL,
    target_entity_id VARCHAR(100),
    entity_type VARCHAR(50) NOT NULL, -- 'customer' or 'category'
    action_description TEXT,
    priority INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
