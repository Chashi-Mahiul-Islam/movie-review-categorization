import review_categorization.rev_categorizer as rc
import click

@click.command()
@click.option("--text", default="Best movie ever!", help="Text to predict")
@click.option("--reviews", default='review_categorization/data/reviews.txt', help="Reviews to train and test")
@click.option("--labels", default='review_categorization/data/labels.txt', help="Labels to train and test")
@click.option("--train", default=False, help="Train mode on")
@click.option("--test", default=False, help="Test mode on")
@click.option("--predict", default=True, help="Predict mode on")
def run(reviews, labels, train, test, predict, text):
    rc.categorize(reviews, labels, train=train, test=test, predict=predict, text=text)
    
if __name__ == "__main__":
    run()