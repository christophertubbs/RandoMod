# RandoMod
A numerical modeling system for generating pseudo-randomized test data

Given baseline time series of what perfect values _should_ look like over a span of time, thresholds of values 
(such as percentiles or definitions of ranges like `low`, `minor`, `major`, `record`, etc), and a guide stating what 
range of values our simulated data will lie in, this system will generate artificial time series with values randomized 
in relation to the `true` values. These randomized values will simulate a series of correct and measurably incorrect 
values when generating said values via normal means (running actual models) may be too time-consuming and cumbersome.