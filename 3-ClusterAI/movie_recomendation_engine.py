"""
This is a movie recommendation engine, that can recommend movies.
Based on ratings given by other people with similar taste.
This program returns 5 movies you will probably enjoy watching and 5 which you probably will not enjoy.
Recommendations are enriched with information from ombd API. Get your key here https://www.omdbapi.com/apikey.aspx 
OMBD api has a daily limit of 1000 request.

https://medium.com/analytics-vidhya/comparative-study-of-the-clustering-algorithms-54d1ed9ea732

Authors:
- Aleksander Stankowski (s27549)
- Daniel Bieliński (s27292)

Environment Setup:

This script requires Python 3.10 or newer (due to type hinting).
It is recommended to use a virtual environment.

1. Create a virtual environment:
  python -m venv venv
2. Activate the environment:
  source venv/bin/activate  (on macOS/Linux)
  .\\venv\\Scripts\\activate   (on Windows)
3. Install required libraries:
  pip install pandas sklearn numpy requests

"""


import pandas
import numpy
import requests
from sklearn.cluster import KMeans

API_BASE_URL = "http://www.omdbapi.com/"
API_KEY = "e2d4b437" #Put your api key here

def load_and_parse_data(filepath: str) -> pandas.DataFrame:
  """
  Loads a CSV file of movie ratings and parses it into a long DataFrame.

  The CSV is expected to be in a wide format (User, Movie1, Rating1, Movie2, Rating2, ...).
  This function transforms it into a long format DataFrame.

  Parameters:
  filepath (str): The path to the movie ratings CSV file.

  Returns:
  pandas.DataFrame: A DataFrame with columns ['user', 'title', 'rating'].
  """
  try:
    raw_dataframe = pandas.read_csv(filepath, header=None, dtype=str, keep_default_na=False)
    print(f"LOG: File {filepath} read")
  except Exception as e:
    print(f"ERROR: Could not load data from {filepath} csv file. {e}")
    exit()

  parsed_data = []

  for _, row in raw_dataframe.iterrows():
    user = row[0]
    if not user or pandas.isna(user):
      continue

    items = row[1:].values

    for i in range(0, len(items) - 1, 2):
      title = items[i]
      rating = items[i+1]

      if not title or pandas.isna(title) or not rating or pandas.isna(rating):
        continue

      try:
        rating = float(rating)
        parsed_data.append(
          {
            "user": user.strip(),
            "title": title.strip(),
            "rating": rating
          }
        )
      except Exception as e:
        print(f"WARNING: Skipping malformed data. User: {user}, Title: {title}, Rating: {rating}. Error: {e}")
        continue

  print(f"LOG: {len(parsed_data)} movie ratings parsed successfully")
  return pandas.DataFrame(parsed_data)

def get_recommendations(df_pivot: pandas.DataFrame, target_user: str) -> None:
  """
  Generates and prints recommendations and anti-recommendations for a target user.

  Finds the user's cluster, calculates movie scores based on cluster averages
  (ignoring 0s/unrated), and prints the top 5 and bottom 5 movies
  the user has not seen.

  Parameters:
  df_pivot (pandas.DataFrame): The pivoted user-movie matrix with a 'cluster' column.
  target_user (str): The name of the user to get recommendations for.
  """
  print(f"\n--- Recommendations for: {target_user} ---")

  try:
    target_user_data = df_pivot.loc[target_user]
  except KeyError:
    print(f"ERROR: User '{target_user}' is not present in CSV file")
    return

  target_user_cluster = int(target_user_data['cluster'])
  print(f"User belongs to cluster: {target_user_cluster}")

  similar_users_df = df_pivot[df_pivot['cluster'] == target_user_cluster].drop(columns=['cluster'])

  print(f"LOG: Found {len(similar_users_df)} users in cluster {target_user_cluster} (including {target_user}):")
  for user in similar_users_df.index:
    print(f"  - {user}")

  similar_users_with_nan = similar_users_df.replace(0, numpy.nan)

  cluster_avg_ratings = similar_users_with_nan.mean()

  unseen_movies_mask = (target_user_data.drop('cluster') == 0)
  unseen_movie_titles = target_user_data.drop('cluster')[unseen_movies_mask].index.tolist()
  print(f"\nLOG: User '{target_user}' has not seen {len(unseen_movie_titles)} movies from the database.")
  unseen_movie_scores = cluster_avg_ratings[unseen_movies_mask]
  scored_unseen_movies = unseen_movie_scores.dropna()
  print(f"LOG: Found {len(scored_unseen_movies)} movies to recommend (rated by cluster, but not by {target_user}).")

  recommendations = scored_unseen_movies.sort_values(ascending=False).head(5)
  print("\n5 movies you might like:")
  if recommendations.empty:
    print("  No recommendations found (your cluster hasn't rated anything you haven't seen).")
  else:
    enrich_and_print_list(recommendations)

  anti_recommendations = scored_unseen_movies.sort_values(ascending=True).head(5)
  print("\n5 movies you might dislike:")
  if anti_recommendations.empty:
    print("  No anti-recommendations found.")
  else:
    enrich_and_print_list(anti_recommendations)

def get_movie_details(title: str) -> dict | None:
  """
  Fetches detailed movie information from the OMDb API by title.

  Uses the global API_KEY and API_BASE_URL constants to make the request.

  Parameters:
  title (str): The title of the movie to search for.

  Returns:
  dict | None: A dictionary containing the movie's API data if found, otherwise None.
  """
  params = {
    "t": title,
    "apikey": API_KEY
  }
  try:
    response = requests.get(API_BASE_URL, params=params)
    response.raise_for_status() # Raise an error for bad responses (4xx, 5xx)
    data = response.json()
    
    # OMDb returns Response 'False' if movie not found
    if data.get("Response") == "True":
      return data
    else:
      print(f"    API could not find '{title}'. Error: {data.get('Error')}")
      return None
  except requests.exceptions.RequestException as e:
    print(f"    LOG: API request failed for '{title}'. Error: {e}")
    return None

def enrich_and_print_list(movie_series: pandas.Series) -> None:
  """
  Prints a list of movies, enriching them with API data if a key is provided.

  If the global API_KEY is not set, it prints a simple list.
  If the API_KEY is set, it calls get_movie_details for each movie and
  prints a detailed, formatted summary.

  Parameters:
  movie_series (pandas.Series): A Series where the index is the movie title
  and the values are the cluster average ratings.
  """
  if not API_KEY:
    for title, score in movie_series.items():
      print(f"  - {title} (Cluster average rating: {score:.2f})")
    return

  for title, score in movie_series.items():
    print(f"\n  - {title} (Cluster average rating: {score:.2f})")
    
    data = get_movie_details(title)
    
    if data:
      print(f"    Year: {data.get('Year', 'N/A')}")
      print(f"    Genre: {data.get('Genre', 'N/A')}")
      print(f"    IMDb Rating: {data.get('imdbRating', 'N/A')}")
      # Truncate plot to 80 characters
      plot = data.get('Plot', 'N/A')
      if len(plot) > 80:
        plot = plot[:80] + "..."
      print(f"    Plot: {plot}")
1
def select_user(user_list: list) -> str:
  """
  Displays a numbered list of users and prompts for a selection.

  Parameters:
  user_list (list): A list of all available user names.

  Returns:
  str: The name of the user that was selected.
  """
  print("\n--- Select a User ---")

  for i, user in enumerate(user_list):
    print(f"  [{i + 1}] {user}")
  
  while True:
    try:
      choice_str = input(f"Enter number (1-{len(user_list)}): ")
      choice_int = int(choice_str)

      if 1 <= choice_int <= len(user_list):
        selected_user = user_list[choice_int - 1]
        return selected_user
      else:
        print(f"  Error: Please enter a number between 1 and {len(user_list)}.")
    except ValueError:
      print("  Error: Please enter a valid number.")

def main() -> None:
  """
  Main entry point for the movie recommendation script.

  1. Load and parse the data.
  2. Pivot the data into a user-movie matrix.
  3. Run K-Means clustering to group users.
  4. Call get_recommendations to print results for the target user.
  """

  DATA_FILE_PATH = r"3-ClusterAI\movie_ratings.csv"

  df = load_and_parse_data(DATA_FILE_PATH)

  print("\nLOG: Pivoting data for clustering...")
  df_pivot = df.pivot_table(index='user', columns='title', values='rating').fillna(0)
  print(f"LOG: Table created {df_pivot.shape} (user x movie_title), rating as value")
  pivot_values = df_pivot.values

  n_clusters = 3
  print(f"\nLOG: Running K-Means (k={n_clusters})...")
  kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(pivot_values)
  labels = kmeans.labels_
  print("LOG: Clustering complete.")

  df_pivot['cluster'] = labels

  all_users = df_pivot.index.tolist()
  target_user = select_user(all_users)

  get_recommendations(df_pivot, target_user)

if __name__ == '__main__':
  main()
