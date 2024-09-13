from StartPage import start_page
from TetrisGame import exit_game_flag, go_to_home_flag


def run():
    start_page()
    print(go_to_home_flag)
    print(exit_game_flag)
    while not exit_game_flag:
        print(go_to_home_flag)
        if go_to_home_flag:
            start_page()


# For testing if running directly
if __name__ == "__main__":
    run()
