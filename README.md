# DBSCAN Clustering API

This project provides a RESTful API for performing DBSCAN clustering analysis on various aspects of a business dataset, including customers, products, suppliers, and country-based sales patterns.

## Features

- Customer behavior analysis
- Product performance analysis
- Supplier performance analysis
- Country-based sales pattern analysis
- Automatic API documentation (Swagger UI and ReDoc)
- RESTful API endpoints
- Async support with FastAPI
- High performance with Uvicorn

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Configure the database connection in `config.py`:
```python
DB_CONFIG = {
    'user': 'your_username',
    'password': 'your_password',
    'host': 'your_host',
    'port': 'your_port',
    'database': 'your_database'
}
```

3. Run the FastAPI application with Uvicorn:
```bash
uvicorn app:app --reload --port 7000
```

## API Endpoints

The API provides the following endpoints:

- `GET /api/customers` - Analyze customer behavior
- `GET /api/products` - Analyze product performance
- `GET /api/suppliers` - Analyze supplier performance
- `GET /api/countries` - Analyze country-based sales patterns

## Documentation

API documentation is automatically available at:
- Swagger UI: `http://localhost:8080/docs`
- ReDoc: `http://localhost:8080/redoc`

## Project Structure

```
.
├── analyses/
│   ├── customer_analysis.py
│   ├── product_analysis.py
│   ├── supplier_analysis.py
│   └── country_analysis.py
├── app.py
├── config.py
├── utils.py
├── requirements.txt
└── README.md
```

## Error Handling

The API includes error handling for:
- Database connection issues
- Invalid parameters
- Processing errors

All errors are returned with appropriate HTTP status codes and error messages.

## Performance

The API is built with FastAPI and Uvicorn, providing:
- High performance
- Async support
- Automatic API documentation
- Type checking
- Input validation 