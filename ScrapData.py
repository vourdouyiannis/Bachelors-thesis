# VOURDOUGIANNIS DIMITRIOS 4326 #

import os
import json
import praw
from praw.models import MoreComments
from nltk.sentiment import SentimentIntensityAnalyzer

# Create an instance
reddit = praw.Reddit(client_id="uvhJFnUWQzUbuWpS2OIDWQ", client_secret="PTog9qkGIPcLH1rzXlyWmrAVHGVc8g", user_agent="vourdouyiannis")

# For subreddit "COVID19"/"atheism"/"Christianity"
# Search the keyword "vaccine"/"deaths"/"jesus"
# Get the top/most relevant 5 posts
subreddit1 = reddit.subreddit("COVID19").search("vaccine", sort="top", limit=5)
#
# subreddit2 = reddit.subreddit("COVID19").search("deaths", sort="top", limit=5)
#
# subreddit3 = reddit.subreddit("atheism").search("jesus", sort="Relevance", limit=5)
#
# subreddit4 = reddit.subreddit("Christianity").search("jesus", sort="Relevance", limit=5)

subreddits = [subreddit1]# , subreddit2, subreddit3, subreddit4]
posts = []


# Process the posts data
def create_posts_dataset(subreddit):
    for post in subreddit:
        post_data = {
            'id': post.id,
            'author': post.author.name,
            'parent_id': "",
            'content': post.title,
            'polarity': "",
        }
        posts.append(post_data)


# Creates the post dataset
for sub in subreddits:
    create_posts_dataset(sub)

posts_filename = 'resources/posts.json'
comments_filename = 'resources/comments.json'

# Save the posts data to a JSON file
with open(posts_filename, 'w') as f:
    json.dump(posts, f)

# Empty the comments file
with open(comments_filename, 'w') as f:
    json.dump([], f)


def analyze(comment):
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Use sentiment analyzer to get polarity score
    score = sia.polarity_scores(comment)

    return score['compound']
    # Determine if the comment is positive or negative based on the polarity score
    # Positive comment
    # if score['compound'] > 0:
    #     return "+"
    # # Negative comment
    # elif score['compound'] < 0:
    #     return "-"
    # # Neutral comment
    # else:
    #     return "0"


def make_tree(submission, post1):
    # Sort the comments into a tree
    tree = []
    comment_stack = submission.comments[:]
    submission.comments.replace_more(limit=None)
    while comment_stack:
        comment = comment_stack.pop(0)
        if isinstance(comment, MoreComments):
            continue
        tree.append(comment)
        comment_stack[0:0] = comment.replies

    # Process the tree's data
    comments = [post1]
    for comment in tree:
        if comment.author is None or comment.author == "AutoModerator":
            continue
        else:
            polarity = analyze(comment.body)
            pid = comment.parent_id.split('_')
            comment_data = {
                'id': comment.id,
                'author': comment.author.name,
                'parent_id': pid[1],
                'content': comment.body,
                'polarity': polarity
            }
            comments.append(comment_data)

    # Store the trees into a JSON file
    # Check if the file exists and is not empty
    if os.path.isfile(comments_filename) and os.path.getsize(comments_filename) > 0:
        # Open the file for reading
        with open(comments_filename, 'r') as f:
            data = json.load(f)

        # Append new data to existing data
        data.append(comments)

        # Write the updated data to the file
        with open(comments_filename, 'w') as f:
            json.dump(data, f)
    else:
        # Write new data to the file
        with open(comments_filename, 'w') as f:
            json.dump([comments], f)


# Load the posts to pick the post we want to make a tree
with open(posts_filename, 'r') as f:
    posts_dict = json.load(f)

# We get the posts to make trees
for i in range(len(posts)):
    # Get submission of each post
    submission = reddit.submission(id=posts_dict[i]['id'])
    # Get the tree for each post
    make_tree(submission, posts_dict[i])
