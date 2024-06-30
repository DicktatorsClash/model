import typer

from rich import print as rprint

import models.classic.v1.classical

app = typer.Typer()


@app.command("train-classic")
def train_classic():
    models.classic.v1.classical.train_model()

@app.command("test-classic")
def sample_func(text: str):
    result = models.classic.v1.classical.test_model(text)
    rprint(result)
    return result

if __name__ == "__main__":
    app()