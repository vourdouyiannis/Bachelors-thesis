
import json
import praw

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
        'content': post.selftext,
    }
    posts.append(post_data)

# Save the posts data to a JSON file
with open('posts.json', 'w') as f:
    json.dump(posts, f)

# Get the first submission (The one with the most upvotes too)
submission = reddit.submission(id=posts[0]['id'])

# Sort the comments into the tree we want
tree = []
comment_stack = submission.comments[:]
submission.comments.replace_more(limit=None)
while comment_stack:
        comment = comment_stack.pop(0)
        tree.append(comment)
        comment_stack[0:0] = comment.replies

# Process the tree's data
comments = []
for comment in tree:
    print(comment.body)  # TODO remove later
    print("################################")  # TODO remove later
    if comment.author is None:
        continue
    else:
        comment_data = {
            'id': comment.id,
            'author': comment.author.name,
            'content': comment.body,
        }
        comments.append(comment_data)

# Store the tree into a JSON file
with open('comments.json', 'w') as f:
    json.dump(comments, f)

# Define that this is the end of the comments.
# From there we can store comments from different post
with open('comments.json', 'r+') as f:
    comments = json.load(f)
    comments.append({'isLast': True})
    f.seek(0)
    json.dump(comments, f)
