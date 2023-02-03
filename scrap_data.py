import praw
from praw.models import MoreComments
import pandas as pd

# Create an instance
reddit = praw.Reddit(client_id="uvhJFnUWQzUbuWpS2OIDWQ", client_secret="PTog9qkGIPcLH1rzXlyWmrAVHGVc8g", user_agent="vourdouyiannis")

# For subreddit COVID19
# Search the keyword "vaccine"
# Get the top 5 posts
subreddit = reddit.subreddit("COVID19").search("vaccine", sort="top", limit=2)

# Collect post's ids in case we need them.
postsID = []
for post in subreddit:
    postsID.append(post.id)

# Get the first submission (The one with the most upvotes too)
submission = reddit.submission(id=postsID[0])

# Get all the comments (CommentForest)
forest = []
ids = []
i = 0
#TODO get the submission in the first element of the list
for comment in submission.comments.list():
    if isinstance(comment, MoreComments):
        continue
    if comment.is_root:
        forest.append([])
        forest[i].append(comment.body)
        ids.append(comment.id)
    else:
        print("here")
        if (comment.parent_id[3:]) in ids:
            print("here")

    if i==27:
        print("here")
    print(str(i)+".", comment.body)
    i += 1

# tree = []
# def get_comments_tree(x=submission.comments):
#     i = 0
#     for comment in submission.comments:
#         tree.append(comment.body)
#         if isinstance(comment, MoreComments):
#             continue
#         print(str(i) + ".", comment.body)
#         i += 1
#     print(tree)
#
# get_comments_tree()
print(forest)
print(ids)

# Converting the comments list into a pandas data frame
# comments_df = pd.DataFrame(comments, columns=['comment'])
# print(comments_df)
