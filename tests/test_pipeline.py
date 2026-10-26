import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.energy_sense import MEASUREMENT_COLUMNS, build_prediction_row, clean_data, load_raw_data, make_supervised, predict_next_power


class PipelineTests(unittest.TestCase):
    def test_loader_parses_timestamp_and_missing_marker(self):
        path=Path(__file__).parent/"_tiny.txt"
        try:
            path.write_text("Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3\n16/12/2006;17:24:00;1.2;0.1;230;5;0;1;2\n16/12/2006;17:25:00;?;0.1;230;5;0;1;2\n",encoding="utf-8")
            frame=load_raw_data(path)
            self.assertTrue(pd.api.types.is_datetime64_any_dtype(frame.DateTime))
            self.assertEqual(frame.loc[0,"Global_active_power"],1.2)
            self.assertTrue(pd.isna(frame.loc[1,"Global_active_power"]))
        finally: path.unlink(missing_ok=True)

    def test_cleaning_and_exact_past_lags(self):
        times=pd.date_range("2020-01-01",periods=40,freq="min")
        frame=pd.DataFrame({"DateTime":times,"Date":times.strftime("%d/%m/%Y"),"Time":times.strftime("%H:%M:%S")})
        for i,col in enumerate(MEASUREMENT_COLUMNS): frame[col]=np.arange(40,dtype=float)+i
        clean=clean_data(frame); x,y,_=make_supervised(clean)
        self.assertEqual(len(x),10)
        self.assertEqual(x.iloc[0].Global_active_power_lag_1,clean.iloc[29].Global_active_power)
        self.assertEqual(x.iloc[0].Global_active_power_lag_30,clean.iloc[0].Global_active_power)
        self.assertEqual(y.iloc[0],clean.iloc[30].Global_active_power)

    def test_validated_prediction_uses_scaler_and_model(self):
        readings=[{col:float(i+j) for j,col in enumerate(MEASUREMENT_COLUMNS)} for i in range(30)]
        class IdentityScaler:
            def transform(self,frame): return frame.to_numpy()
        class DummyModel:
            def predict(self,x,verbose=0): return np.array([[0.25]])
        row=build_prediction_row(readings,"2020-01-01 12:30")
        self.assertEqual(row.shape,(1,12))
        self.assertEqual(predict_next_power(DummyModel(),IdentityScaler(),readings,"2020-01-01 12:30",0.5),readings[-1]["Global_active_power"]+0.125)
        with self.assertRaisesRegex(ValueError,"30 consecutive"):
            build_prediction_row(readings[:5])


if __name__=="__main__": unittest.main()
