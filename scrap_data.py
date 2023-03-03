# VOURDOUGIANNIS DIMITRIOS 4326 #

import praw
from praw.models import MoreComments
from anytree import Node, RenderTree
import pandas as pd

# Create an instance
reddit = praw.Reddit(client_id="uvhJFnUWQzUbuWpS2OIDWQ", client_secret="PTog9qkGIPcLH1rzXlyWmrAVHGVc8g", user_agent="vourdouyiannis")

# For subreddit COVID19
# Search the keyword "vaccine"
# Get the top 5 posts
subreddit = reddit.subreddit("COVID19").search("vaccine", sort="top", limit=1)

# Collect post's ids in case we need them.
postsID = []
for post in subreddit:
    postsID.append(post.id)

# Get the first submission (The one with the most upvotes too)
submission = reddit.submission(id=postsID[0])

# Get a tree with all comments in depth-first search
tree = []

comment_stack = submission.comments[:]
submission.comments.replace_more(limit=None)
while comment_stack:
        comment = comment_stack.pop(0)
        tree.append(comment)
        # if len(comment.replies) == 0:
        #     tree.append("end")
        # else:
        #     tree.append("more")
        comment_stack[0:0] = comment.replies

for comment in tree:
    print(comment.body)
    print("################################")
