if __name__ == "__main__":
    import uvicorn
    uvicorn.run("summarization_api.main:app", host="0.0.0.0", port=8000, reload=True)
