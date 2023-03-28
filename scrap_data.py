# VOURDOUGIANNIS DIMITRIOS 4326 #

import json
import praw
from praw.models import MoreComments
import os

# Create an instance
reddit = praw.Reddit(client_id="uvhJFnUWQzUbuWpS2OIDWQ", client_secret="PTog9qkGIPcLH1rzXlyWmrAVHGVc8g", user_agent="vourdouyiannis")

# For subreddit COVID19
# Search the keyword "vaccine"
# Get the top 5 posts
subreddit = reddit.subreddit("COVID19").search("vaccine", sort="top", limit=5)

# Process the posts data
posts = []
for post in subreddit:
    post_data = {
        'id': post.id,
        'title': post.title,
        'author': post.author.name,
    }
    posts.append(post_data)

posts_filename = 'resources/posts.json'
comments_filename = 'resources/comments.json'

# Save the posts data to a JSON file
with open(posts_filename, 'w') as f:
    json.dump(posts, f)

# Load the posts to pick the post we want to make a tree
with open(posts_filename, 'r') as f:
    posts_dict = json.load(f)

# Empty the comments file
with open(comments_filename, 'w') as f:
    json.dump([], f)
def make_tree(submission):
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
    comments = []
    for comment in tree:
        if comment.author is None or comment.author == "AutoModerator":
            continue
        else:
            pid = comment.parent_id.split('_')
            comment_data = {
                'id': comment.id,
                'author': comment.author.name,
                'parent_id': pid[1],
                'content': comment.body,
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

# We get 5 posts to make trees
for i in range(5):
    # Get submission of each post
    submission = reddit.submission(id=posts_dict[i]['id'])
    # Get the tree for each post
    make_tree(submission)
