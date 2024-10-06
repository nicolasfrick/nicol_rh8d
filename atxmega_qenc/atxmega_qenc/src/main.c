#include <stdio.h>
#include <asf.h>

// PIN 2,3
#define QDEC1 (1<<2)
#define QDEC2 (1<<3)
// resolution
#define QUADRATURE_PPR 2048  // AMT10 max resolution
#define QUADRATURE_COUNTS (QUADRATURE_PPR * 4)  // 8192 counts per revolution
#define INIT_COUNTS (QUADRATURE_PPR * 2) // 4096 
// sampling filter
#define FILTER_SAMPLES EVSYS_DIGFILT_2SAMPLES_gc
// binary data
#define START_BYTE 0xAA
#define END_BYTE   0xBB
#define ESCAPE_BYTE 0x7D

void initUart(void);
void sendMsg(char* data);
void initLed(void);
uint8_t getChar(void);
void putChar(uint8_t chr);
void resetPosition(void);
void initDistalQdec(void);
void initMedialQdec(void);
void initProximalQdec(void);
void sendByte(uint8_t data);
void sendEscByte(uint8_t data);
void sendData(uint16_t num1, uint16_t num2, uint16_t num3);

// handle usb request
ISR(USARTE0_RXC_vect)
{
	// LED on
	PORTA.OUT |= (1 << 7);
	uint8_t data = USARTE0.DATA;
	if (data == START_BYTE) 
		// prox_pos, med_pos, dist_pos
		sendData(TCC1.CNT, TCD0.CNT, TCD1.CNT);
	else if (data == END_BYTE)
		resetPosition();
	else
		sendMsg("Send 0xBB for counter reset, 0xAA for position query\n");
	// LED off
	PORTA.OUT &= (0 << 7);
}

int main (void)
{
	sei();
	
	initLed();
	initUart();
	initProximalQdec();
	initMedialQdec();
	initDistalQdec();

	while (1)
	{
		// char msg[200];
		// uint16_t prox_pos = TCC1.CNT;
		// uint16_t med_pos = TCD0.CNT;
		// uint16_t dist_pos = TCD1.CNT;
		// sprintf(msg, "%d %d %d\n", prox_pos, med_pos, dist_pos);
		// sendMsg(msg);
		// delay_ms(250);
	}
}

void resetPosition(void)
{
	TCD1.CNT = INIT_COUNTS;
	TCC1.CNT = INIT_COUNTS;
	TCD0.CNT = INIT_COUNTS;
}

void initProximalQdec(void)
{
	// PORTC, CHANNEL0, PIN2,3, TCC1
	PORTC.DIRCLR = QDEC1 | QDEC2;
	PORTC.PIN1CTRL = PORT_INVEN_bm;
	PORTC.PIN2CTRL = PORT_ISC_LEVEL_gc | PORT_OPC_PULLUP_gc;
	PORTC.PIN3CTRL = PORT_ISC_LEVEL_gc | PORT_OPC_PULLUP_gc;
	EVSYS.CH0MUX = EVSYS_CHMUX_PORTC_PIN2_gc;
	EVSYS.CH0CTRL = EVSYS_QDEN_bm | FILTER_SAMPLES;
	TCC1.CTRLD = TC_EVACT_QDEC_gc | TC_EVSEL_CH0_gc;
	TCC1.PER = QUADRATURE_COUNTS;
	TCC1_CTRLA = TC_CLKSEL_DIV1_gc;
	TCC1.CNT = INIT_COUNTS;
}

void initMedialQdec(void)
{ 
	// PORTD, CHANNEL2, PIN2,3, TCD0
	PORTD.DIRCLR = QDEC1 | QDEC2; // 2,3 input 
	PORTD.PIN1CTRL = PORT_INVEN_bm; // logic invert
	PORTD.PIN2CTRL = PORT_ISC_LEVEL_gc | PORT_OPC_PULLUP_gc; //  level sensitive, pullup ena
	PORTD.PIN3CTRL = PORT_ISC_LEVEL_gc | PORT_OPC_PULLUP_gc; //  level sensitive, pullup ena
	EVSYS.CH2MUX = EVSYS_CHMUX_PORTD_PIN2_gc; // set up channel 2 to use the signal from PORTD pin 2 as input
	EVSYS.CH2CTRL = EVSYS_QDEN_bm | FILTER_SAMPLES; // ena qdec and set filtering for channel 2
	TCD0.CTRLD = TC_EVACT_QDEC_gc | TC_EVSEL_CH2_gc; // conf timer for qdec event from channel 2
	TCD0.PER = QUADRATURE_COUNTS; // set encoder PPR
	TCD0_CTRLA = TC_CLKSEL_DIV1_gc; // high speed count
	TCD0.CNT = INIT_COUNTS; // zero counter
}

void initDistalQdec(void)
{
	// PORTA, CHANNEL4, PIN2,3, TCD1
	PORTA.DIRCLR = QDEC1 | QDEC2;
	PORTA.PIN1CTRL = PORT_INVEN_bm;
	PORTA.PIN2CTRL = PORT_ISC_LEVEL_gc | PORT_OPC_PULLUP_gc;
	PORTA.PIN3CTRL = PORT_ISC_LEVEL_gc | PORT_OPC_PULLUP_gc;
	EVSYS.CH4MUX = EVSYS_CHMUX_PORTA_PIN2_gc;
	EVSYS.CH4CTRL = EVSYS_QDEN_bm | FILTER_SAMPLES;
	TCD1.CTRLD = TC_EVACT_QDEC_gc | TC_EVSEL_CH4_gc;
	TCD1.PER = QUADRATURE_COUNTS;
	TCD1_CTRLA = TC_CLKSEL_DIV1_gc;
	TCD1.CNT = INIT_COUNTS;
}

void initLed(void)
{
	PORTA.DIR |= (1 << 7);
	PORTA.OUT &= (0 << 7);	
}

void initUart(void)
{
	/*
	Initialize the USART
		-> 19200 Baud @ 2 MHz with CLK2X = 0, BSCALE = -5
		-> Rx InterruptW
		-> Use Rx and Tx
		-> 8N1
	*/
	USARTE0.BAUDCTRLA = 0xB0 & 0xFF;
	USARTE0.BAUDCTRLB = ((0xB0 & 0xF00) >> 0x08);
	USARTE0.BAUDCTRLB |= ((-5 & 0x0F) << 0x04);
	USARTE0.CTRLA = USART_RXCINTLVL_LO_gc;
	USARTE0.STATUS |= USART_RXCIF_bm;
	USARTE0.CTRLB = USART_TXEN_bm | USART_RXEN_bm;
	USARTE0.CTRLC = USART_CHSIZE_8BIT_gc;
	USARTE0.CTRLC &= ~(USART_PMODE0_bm | USART_PMODE1_bm | USART_SBMODE_bm);
	PORTE.DIR = 0x08;
	
	while(!(USARTE0.STATUS & USART_DREIF_bm));
	
	PMIC.CTRL = PMIC_LOLVLEN_bm;
	sei();
}

void sendMsg(char* data)
{
	while(*data)
	{
		while(!(USARTE0.STATUS & USART_DREIF_bm));
		USARTE0.DATA = *data++;
	}
}

uint8_t getChar(void)
{	
	// blocking
	while (!(USARTE0.STATUS & USART_RXCIF_bm));
	return ((uint8_t)(&USARTE0)->DATA);
}

void putChar(uint8_t chr)
{
	while(!(USARTE0.STATUS & USART_DREIF_bm));
	USARTE0.DATA = chr;
	while(!(USARTE0.STATUS & USART_DREIF_bm));
	USARTE0.DATA = '\n';
}

void sendByte(uint8_t data) 
{
	while(!(USARTE0.STATUS & USART_DREIF_bm));
	USARTE0.DATA = data;
}

void sendEscByte(uint8_t data) 
{
	if (data == START_BYTE || data == END_BYTE || data == ESCAPE_BYTE) 
	{
		// send escape byte
		sendByte(ESCAPE_BYTE);
		// send xored data
		sendByte(data ^ 0x20);
	} 
	else 
	{
		sendByte(data);
	}
}

void sendData(uint16_t num1, uint16_t num2, uint16_t num3) 
{
	sendByte(START_BYTE);
	sendEscByte((uint8_t)(num1 >> 8));   // high byte
	sendEscByte((uint8_t)(num1 & 0xFF)); // low byte 
	sendEscByte((uint8_t)(num2 >> 8));   // high byte
	sendEscByte((uint8_t)(num2 & 0xFF)); // low byte 
	sendEscByte((uint8_t)(num3 >> 8));   // high byte
	sendEscByte((uint8_t)(num3 & 0xFF)); // low byte 
	sendByte(END_BYTE);
}
