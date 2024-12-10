import matplotlib.pyplot as plt
from IPython import display

# Turn on interactive mode for matplotlib
plt.ion()

def plot(scores, mean_scores):
    # Clear the current output and display the updated plot
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    # Plot the scores
    plt.plot(scores)
    # Plot the mean scores
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    # Annotate the last score
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    # Annotate the last mean score
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    # Show the plot without blocking the execution
    plt.show(block=False)
    # Pause to update the plot
    plt.pause(0.1)