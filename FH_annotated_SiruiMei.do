* Feldstein-Horioka Puzzle: Savings-Investment Correlation Analysis
* Sirui (Silvie) Mei
* Johns Hopkins SAIS - Quantitative Global Economics

* ------------------------------------------------------------
* install estout to store and export regression outputs
ssc install estout, replace

* reshape from wide to long format for panel analysis
reshape long i_ s_, i(country) j(year)
rename i_ i
rename s_ s

* restrict to G7 as example subsample -- modify as needed
* note: this modifies the dataset in memory, save a copy first
keep if inlist(country, "Canada", "France", "Germany", "Italy", "Japan", "United Kingdom", "United States")

* ------------------------------------------------------------
* panel setup
destring year, replace
encode country, gen(country_id)
xtset country_id year

* ------------------------------------------------------------
* baseline pooled OLS -- replicates original F-H (1980) specification
* beta close to 1 = low capital mobility; close to 0 = high mobility
reg i s
eststo

* display stored results
esttab
* suppress year FE coefficients for cleaner output
esttab, indicate("Time Fixed Effects=*.year")

eststo clear

* ------------------------------------------------------------
* within-estimator for 1990s -- controls for time-invariant country factors
xtreg i s if year>=1990 & year<2000, fe
eststo

* time FE only -- absorbs global shocks common to all countries
reg i s i.year
eststo

* two-way FE: controls for both country heterogeneity and time trends
* most conservative specification
xtreg i s i.year, fe
eststo

* export all results
esttab using FH_results.csv

* ------------------------------------------------------------
* cross-sectional approach following original F-H method
* collapse to country averages over selected period
* note: modifies dataset -- save copy before running
collapse i s if year>=1990 & year<2000, by(country)
reg i s
