class Document:
    def __init__(self, 
                 content: str,
                 id: str = None):
        # metadata can be added further down (date of creation, author, modified date, etc.)
        self.content = content
        self.id = id

    def __str__(self):
        return f"{self.content}"
    
    def __repr__(self):
        return f"{self.content}"