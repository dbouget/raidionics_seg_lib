#!/bin/bash
isort --sl raidionicsseg
black --line-length 120 raidionicsseg
flake8 --max-line-length 120 --ignore "E203" raidionicsseg