# install.packages("reshape2")
library(ggplot2)
library(patchwork)
library(dplyr)
library(reshape2)
library(ctsmTMB)

# Get content into a data frame
data <- read.csv("ctsm/EnergyPrices/priceData.csv",
                header = TRUE, sep = ";")
     

# Remove nan values
delete.na <- function(data, n=0) {
  data[rowSums(is.na(data)) <= n,]
}

df <- delete.na(data)
df_time <- df[,'t']
time <- seq(0,length(df_time)-1,by=1)
spot <- abs(df[,'Spot.price'])


#Extract a subset of observations
# num_obs = 1000
# ids = seq(1,length(df_time),by=round(length(df_time) / num_obs))
# time_obs = time[ids]
# spot_obs = spot[ids]
# plot(time_obs,spot_obs)


# Simulate data using Euler Maruyama
set.seed(20)
pars = c(theta=10, mu=mean(spot), sigma_x=sd(spot), sigma_y=0.1)

dt.sim = 1
t.sim = time
dw = rnorm(length(t.sim)-1,sd=sqrt(dt.sim))
u.sim = cumsum(rnorm(length(t.sim),sd=0.05))

x = spot
t.obs = time
y = spot

# # Extract observations and add noise
# t.obs = t.sim[ids]
# # forcing input
# u = u.sim[ids]
# dw = dw[ids]
# t.sim = t.sim[ids]

# Create data
.data = data.frame(
  t = time,
  y = spot,
  u = u.sim
)


############################################################
# Model creation and estimation
############################################################

# Create model object
model = ctsmTMB$new()

# Set name of model (and the created .cpp file)
model$setModelname("ornstein_uhlenbeck")

# Add system equations
model$addSystem(
  dx ~ theta * (mu-x+u) * dt + sigma_x*dw
)
# Add observation equations
model$addObs(
  y ~ x
)

# Set observation equation variances
model$setVariance(
  y ~ sigma_y^2
)

# Specify algebraic relations
model$setAlgebraics(
  theta   ~ exp(logtheta),
  sigma_x ~ exp(logsigma_x),
  sigma_y ~ exp(logsigma_y)
)

# Add vector input
model$addInput(u)

# Specify parameter initial values and lower/upper bounds in estimation
model$setParameter(
  logtheta    = log(c(initial = 1, lower=1e-5, upper=50)),
  mu          = c(initial=1.5, lower=0, upper=5),
  logsigma_x  = log(c(initial=1, lower=1e-10, upper=30)),
  logsigma_y  = log(c(initial=1e-1, lower=1e-10, upper=30))
)

# Set initial state mean and covariance
model$setInitialState(list(x[1], 1e-1*diag(1)))

# Carry out estimation with default settings (extended kalman filter)
fit <- model$estimate(data=.data, method="ekf")

# Check parameter estimates against truth
p0 = fit$par.fixed
cbind(c(exp(p0[1]),p0[2],exp(p0[3]),exp(p0[4])), pars)

# Create plot of one-step predictions, simulated states and observations
t.est = fit$states$mean$prior$t
x.mean = fit$states$mean$prior$x
x.sd = fit$states$sd$prior$x
plot1 = ggplot() +
  geom_ribbon(aes(x=t.est, ymin=x.mean-2*x.sd, ymax=x.mean+2*x.sd),fill="grey", alpha=0.9) +
  geom_line(aes(x=t.est, x.mean),col="steelblue",lwd=1) +
  geom_line(aes(x=t.sim,y=x)) + 
  geom_point(aes(x=t.obs,y=y),col="tomato",size=1) +
  labs(title="1-Step State Estimates vs Observations", x="Time", y="") +
  theme_minimal()

length(y)
plot(plot1)

# Predict to obtain k-step-ahead predictions to see model forecasting ability
pred.list = model$predict(data=.data, 
                        k.ahead=10, 
                        method="ekf",
)

# Create plot all 10-step predictions against data
pred = pred.list$states
pred10step = pred %>% dplyr::filter(k.ahead==10)
plot2 = ggplot() +
  geom_ribbon(aes(x=pred10step$t.j, 
                  ymin=pred10step$x-2*sqrt(pred10step$var.x),
                  ymax=pred10step$x+2*sqrt(pred10step$var.x)),fill="grey", alpha=0.9) +
  geom_line(aes(x=pred10step$t.j,pred10step$x),color="steelblue",lwd=1) +
  geom_point(aes(x=t.obs,y=y),color="tomato",size=1) +
  labs(title="10 Step Predictions vs Observations", x="Time", y="") +
  theme_minimal()

pred10step$var.x
plot(plot2)

# Perform full prediction without data update
pred.list = model$predict(data=.data, 
                        k.ahead=1e6, 
                        method="ekf",
)

# Perform full simulation without data update
sim.list = model$simulate(data=.data, 
                        k.ahead=1e6, 
                        method="ekf"
)

# Collapse simulation data for easy use with ggplot 
sim.df = sim.list$states$x$i0 %>%
  select(!c("i","j","t.i","k.ahead")) %>%
  reshape2::melt(., id.var="t.j")

# Plot all full simulations and the full prediction against observations
# (full means no data-update at all)
plot3 = ggplot() +
  geom_line(data=sim.df, aes(x=t.j, y=value, group=variable),color="grey") +
  geom_line(aes(x=pred.list$states$t.j,y=pred.list$states$x),color="steelblue") +
  geom_point(aes(x=t.obs,y=y),color="tomato",size=1) +
  labs(title="No Update Prediction and Simulations vs Observations", x="Time", y="") +
  theme_minimal() + theme(legend.position = "none")

plot(plot3)
# Draw both plots
patchwork::wrap_plots(plot1, plot2, plot3, ncol=1)
# Plot one-step-ahead residual analysis using the command below
plot(fit)

