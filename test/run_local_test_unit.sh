#!/bin/bash
pytest --cov-report=html --cov=ard test/unit

rm -rf test/unit/layout/problem*_out

#
