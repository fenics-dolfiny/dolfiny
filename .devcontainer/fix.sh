#! /usr/bin/bash

chown vscode:vscode -R /dolfinx-env

ln -sv $(uname -m)-linux-gnu-g++ /usr/bin/$(uname -m)-linux-gnu-g++-11
