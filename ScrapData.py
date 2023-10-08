# VOURDOUGIANNIS DIMITRIOS 4326 #

import json
import praw
from praw.models import MoreComments
from nltk.sentiment import SentimentIntensityAnalyzer
import prawcore
import time

# Create an instance
reddit = praw.Reddit(client_id="uvhJFnUWQzUbuWpS2OIDWQ", client_secret="PTog9qkGIPcLH1rzXlyWmrAVHGVc8g", user_agent="vourdouyiannis")

# For subreddit "COVID19"/"atheism"/"Christianity"
# Search the keyword "vaccine"/"deaths"/"jesus"
# Get the top/most relevant 5 posts
subreddit1 = reddit.subreddit("COVID19").search("vaccine", sort="top", limit=500)  # 140 acceptable posts

subreddit2 = reddit.subreddit("COVID19").search("deaths", sort="top", limit=500)  # 54 acceptable posts

subreddit3 = reddit.subreddit("atheism").search("jesus", sort="Relevance", limit=500)  # 217 acceptable posts

subreddit4 = reddit.subreddit("Christianity").search("jesus", sort="Relevance", limit=500)  # 207 acceptable posts

subreddits = [subreddit1, subreddit2, subreddit3, subreddit4]
posts = []


def count_all_comments(comments):
    total_count = 0
    for comment in comments:
        total_count += 1
        total_count += count_all_comments(comment.replies)
    return total_count

def create_posts_dataset(subreddit, comment_type):
    rejected = 0
    accepted = 0
    for post in subreddit:
        retry_count = 3
        while retry_count > 0:
            try:
                post = reddit.submission(id=post.id)
                post.comments.replace_more(limit=20)
                num_comments = count_all_comments(post.comments)
                if num_comments < 50:
                    print(num_comments, ": rejected")
                    rejected += 1
                    break
                else:
                    print(num_comments, ": accepted")
                    accepted += 1
                    post_data = {
                        'id': post.id,
                        'author': post.author.name,
                        'parent_id': "",
                        'content': post.title,
                        'sentiment': "",
                        'type': comment_type  # Save the type of comment
                    }
                    posts.append(post_data)
                    break
            except prawcore.exceptions.TooManyRequests:
                print("Hit rate limit, waiting for 10 seconds before retrying...")
                time.sleep(10)
                retry_count -= 1
            except Exception as e:
                print(f"Unexpected error: {e}. Retrying...")
                retry_count -= 1

    print("Number of accepted: ", accepted)
    print("Number of rejected: ", rejected)

# Provide the appropriate type for each subreddit
create_posts_dataset(subreddit1, "non_controversial")
create_posts_dataset(subreddit2, "non_controversial")
create_posts_dataset(subreddit3, "controversial")
create_posts_dataset(subreddit4, "controversial")

posts_filename = 'resources/posts.json'
comments_filename = 'resources/comments_non_controversial.json'
comments_filename2 = 'resources/comments_controversial.json'

# Save the posts data to a JSON file
with open(posts_filename, 'w') as f:
    json.dump(posts, f)

# Empty the comments file
with open(comments_filename, 'w') as f:
    json.dump([], f)

# Empty the comments file
with open(comments_filename2, 'w') as f:
    json.dump([], f)

all_comments_non_controversial = []
all_comments_controversial = []


def analyze(comment):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(comment)
    return score['compound']


def make_tree(submission, post1):
    tree = []
    comment_stack = submission.comments[:]
    submission.comments.replace_more(limit=100)
    count = 0
    while comment_stack and count < 2000:
        comment = comment_stack.pop(0)
        if isinstance(comment, MoreComments):
            continue
        tree.append(comment)
        comment_stack[0:0] = comment.replies
        count += 1

    comments = [post1]
    for comment in tree:
        if comment.author is None or comment.author == "AutoModerator":
            continue
        else:
            sentiment = analyze(comment.body)
            pid = comment.parent_id.split('_')
            comment_data = {
                'id': comment.id,
                'author': comment.author.name,
                'parent_id': pid[1],
                'content': comment.body,
                'sentiment': sentiment
            }
            comments.append(comment_data)

    # Append the comments to the appropriate in-memory list after processing all comments
    if post1["type"] == "non_controversial":
        all_comments_non_controversial.append(comments)
    else:
        all_comments_controversial.append(comments)


with open('resources/posts.json', 'r') as f:
    posts_dict = json.load(f)

for i in range(len(posts)):
    submission = reddit.submission(id=posts_dict[i]['id'])
    try:
        make_tree(submission, posts_dict[i])
    except prawcore.exceptions.TooManyRequests:
        print("Hit rate limit, waiting for 10 seconds before retrying...")
        time.sleep(10)

with open('resources/comments_non_controversial.json', 'w') as f:
    json.dump(all_comments_non_controversial, f)
with open('resources/comments_controversial.json', 'w') as f:
    json.dump(all_comments_controversial, f)
