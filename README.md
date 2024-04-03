# Time Series Analysis and Forecasting: Electricity Consumption in Brazil (WIP)

## Introduction
Energy planning plays a critical role in the evolution of the electricity supply system in Brazil, covering, among other aspects, the sizing and specification of new lines and power transmission substations. A mismatch or lack of precision in this planning, in terms of meeting domestic demand, can lead to a significant increase in costs for the end consumer. This increase in costs can occur due to the need to resort to more expensive energy production methods or to the importation of this input from neighboring countries.

With this context in mind, this project aims to address one of the most significant challenges in Brazil's energy planning: the difficulty in predicting electricity consumption. The objective is to assess consumption trends and identify external factors that may influence this consumption. Additionally, the project proposes to test different forecasting models to find the one that best adapts to the prediction of Brazil's electric energy consumption.


## Data
Main database:
Database extracted from the website of the Energy Research Company [EPE], with data on Monthly Electricity Consumption by Class (regions and subsystems), for the period from 2004 to 2022.
Available at: <https://www.epe.gov.br/pt/publicacoes-dados-abertos/publicacoes/consumo-de-energia-eletrica>

Description of the secondary databases:
- Industrial Production Level Indicator [IPL]
  The Industrial Production Level Index is derived from the data of the Monthly Industrial Production Physical Survey [PIM-PF], conducted by the Brazilian Institute of Geography and Statistics (IBGE). This survey considers the Industrial Transformation Value and the Gross Value of Industrial Production for the selection of activities and products, along with their respective producers, to be evaluated. The index calculation is performed using the Laspeyres method of chain base fixing, and the segments and products not included in the calculation have their weights distributed proportionally to those that are represented (IBGE, 2023).
  Database extracted from the Central Bank of Brazil's Time Series Management System, with data originating from the IBGE.
  Available at: <https://www3.bcb.gov.br/sgspub/> 

- Monthly Gross Domestic Product of Brazil [GDP]
  The GDP is an economic indicator that quantifies the wealth production of a country during a specific period. Its calculation involves summing up all the final goods and services generated by economic agents residing in a nation. The GDP plays a fundamental role in assessing a country's economic health (IBGE, 2023).
  Database extracted from the website of the Institute of Applied Economic Research [Ipea].
  Available at: <http://www.ipeadata.gov.br/ExibeSerie.aspx?serid=521274780&module=M>

- Capacity Utilization [UIC]
  Installed capacity refers to the maximum amount of production that a company or sector can achieve using available resources. The capacity utilization indicator is expressed in percentage format and is calculated by the ratio of effective production to maximum production capacity (CNI, 2023).
  Database extracted from the National Industry Confederation (CNI) indicators system.
  Available at: <https://indicadores.sistemaindustria.com.br/indicadores/externo/consultarResultados.faces#>


## Metrics


## Conclusion
