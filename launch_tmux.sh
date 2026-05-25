#!/bin/bash

# Start a new tmux session named 'my_session'
tmux new-session -d -s my_session

# Split the window into four panes
tmux split-window -h -t my_session

tmux split-window -v -t my_session:0.0
tmux split-window -v -t my_session:0.1

tmux select-layout tiled

#tmux resize-pane -t 0 -R 20
tmux resize-pane -t my_session:0.0 -R 20

# Set up placeholders for the scripts to run in each pane
tmux send-keys -t my_session:0.0 'htop;echo "panel 0.0"' C-m
tmux send-keys -t my_session:0.1 'nvtop;clear;echo "panel 0.1"' C-m
tmux send-keys -t my_session:0.2 'clear;echo "panel 0.2"' C-m
tmux send-keys -t my_session:0.3 'clear;echo "panel 0.3"' C-m

# Enable mouse mode
tmux set-option -g mouse on

# Attach to the session
tmux attach-session -t my_session
